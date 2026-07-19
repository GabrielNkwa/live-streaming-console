from flask import Flask, render_template, request, Response, session, redirect, url_for, jsonify, abort, send_from_directory
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import cv2
from ultralytics import YOLO
import threading
import time
import logging
import numpy as np
import os
import platform
import secrets
import sys
import re
from datetime import timedelta
from urllib.parse import urlparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("Starting Flask app initialization")

def bundle_dir():
    """Directory containing read-only bundled resources (the model file,
    templates/, static/). Under a PyInstaller build this is the extracted
    bundle directory (sys._MEIPASS), not the current working directory -
    otherwise a relative path like 'yolo11n.pt' only resolves correctly if
    the app happens to be launched from that exact directory."""
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))

# root_path is passed explicitly (rather than relying on Flask's own
# __file__-based auto-detection, which can be unreliable inside a frozen
# PyInstaller bundle) so templates/static are found correctly either way -
# bundle_dir() already returns the same value Flask would have auto-detected
# when not frozen, so this changes nothing for normal/Docker runs.
#
# instance_path is set by the desktop entrypoint (desktop.py) to an
# OS-appropriate, writable, per-user app-data directory (e.g.
# %APPDATA%\DLC Surveillance). Left unset for Docker/server deployments,
# where Flask's normal <root>/instance default (bind-mounted in Docker) is
# correct as-is.
app = Flask(
    __name__,
    static_url_path=None,
    root_path=bundle_dir(),
    instance_path=os.environ.get('APP_INSTANCE_PATH'),
)

def _load_or_create_secret_key():
    env_key = os.environ.get('FLASK_SECRET_KEY')
    if env_key:
        return env_key
    # No explicit key configured - persist a generated one to disk instead of
    # regenerating on every restart, which would otherwise silently log out
    # every user (and invalidate every CSRF token) each time the process
    # restarts. This is a reasonable default for a single-instance
    # self-hosted deployment; set FLASK_SECRET_KEY explicitly if you ever
    # run more than one instance/replica sharing the same database.
    os.makedirs(app.instance_path, exist_ok=True)
    key_path = os.path.join(app.instance_path, '.secret_key')
    if os.path.exists(key_path):
        with open(key_path, 'r') as f:
            existing = f.read().strip()
        if existing:
            logger.info("Loaded persistent secret key from %s", key_path)
            return existing
    generated = secrets.token_urlsafe(32)
    with open(key_path, 'w') as f:
        f.write(generated)
    try:
        os.chmod(key_path, 0o600)
    except OSError:
        pass
    logger.warning("Generated a new persistent secret key at %s. Set FLASK_SECRET_KEY explicitly in production.", key_path)
    return generated

app.secret_key = _load_or_create_secret_key()

app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    SESSION_COOKIE_SECURE=os.environ.get('FLASK_ENV', 'production') != 'development',
    PERMANENT_SESSION_LIFETIME=timedelta(minutes=20),
    PREFERRED_URL_SCHEME='https',
    SQLALCHEMY_DATABASE_URI=os.environ.get('DATABASE_URL', 'sqlite:///dlc.db'),
    SQLALCHEMY_TRACK_MODIFICATIONS=False
)

if os.environ.get('USE_PROXY_FIX') == '1':
    app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

ASSETS_FOLDER = os.path.join(app.root_path, 'static', 'assets')
os.makedirs(ASSETS_FOLDER, exist_ok=True)

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'index'

logger.info("Loading YOLO model")
try:
    model = YOLO(os.path.join(bundle_dir(), 'yolo11n.pt'))
    logger.info("YOLO model loaded successfully")
except Exception as e:
    logger.error(f"Error loading YOLO model: {e}")
    model = None

camera_lock = threading.Lock()
active_cameras = {}          # url -> cv2.VideoCapture, only ever mutated while holding camera_lock
camera_last_used = {}        # url -> time.monotonic() of last successful use, guarded by camera_lock
camera_url_locks = {}        # url -> threading.Lock(), one per distinct camera URL, guarded by camera_lock
MAX_ACTIVE_CAMERAS = int(os.environ.get('MAX_ACTIVE_CAMERAS', 10))
MAX_CAMERAS_PER_USER = int(os.environ.get('MAX_CAMERAS_PER_USER', 20))
CAMERA_CONNECT_TIMEOUT_MS = int(os.environ.get('CAMERA_CONNECT_TIMEOUT_MS', 8000))
CAMERA_READ_TIMEOUT_MS = int(os.environ.get('CAMERA_READ_TIMEOUT_MS', 5000))
CAMERA_IDLE_TIMEOUT_SECONDS = int(os.environ.get('CAMERA_IDLE_TIMEOUT_SECONDS', 120))
CAMERA_EVICTION_GRACE_SECONDS = 2

login_lock = threading.Lock()
failed_logins = {}
MAX_LOGIN_ATTEMPTS = 5
LOGIN_LOCKOUT_SECONDS = 300

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(20), nullable=False, default='user')
    cameras = db.relationship('Camera', backref='owner', cascade='all, delete-orphan', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Camera(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    owner_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name = db.Column(db.String(64), nullable=False)
    url = db.Column(db.String(200), nullable=False)

    __table_args__ = (db.UniqueConstraint('owner_id', 'name', name='uq_camera_owner_name'),)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def generate_csrf_token():
    token = session.get('csrf_token')
    if not token:
        token = secrets.token_urlsafe(32)
        session['csrf_token'] = token
    return token

def is_safe_camera_name(camera_name):
    return bool(re.fullmatch(r"[A-Za-z0-9 _\-]{1,64}", camera_name))

def is_valid_username(username):
    return bool(re.fullmatch(r"[A-Za-z0-9_.\-]{3,32}", username))

def is_valid_password(password):
    return isinstance(password, str) and len(password) >= 8

def is_safe_camera_url(camera_url):
    if not camera_url or len(camera_url) > 200:
        return False
    if camera_url.startswith('webcam:'):
        try:
            index = int(camera_url.split(':', 1)[1])
            return 0 <= index <= 9
        except ValueError:
            return False
    parsed = urlparse(camera_url)
    if parsed.scheme not in ('rtsp', 'http', 'https'):
        return False
    if not parsed.hostname:
        return False
    hostname = parsed.hostname.lower()
    # Cameras are now added by any authenticated user, so block loopback/
    # unspecified targets to reduce SSRF risk against the server itself.
    # LAN camera ranges (192.168.x.x, 10.x.x.x, etc.) are intentionally
    # still allowed since that's where real RTSP cameras live.
    if hostname in ('127.0.0.1', 'localhost', '0.0.0.0', '::1'):
        return False
    if hostname.startswith('127.') or hostname.startswith('169.254.'):
        return False
    return True

def is_admin():
    return current_user.is_authenticated and current_user.role == 'admin'

def admin_count(exclude_user_id=None):
    query = User.query.filter_by(role='admin')
    if exclude_user_id is not None:
        query = query.filter(User.id != exclude_user_id)
    return query.count()

def mask_camera_url(url):
    if not url or url.startswith('webcam:'):
        return url
    parsed = urlparse(url)
    if not parsed.hostname:
        return url
    netloc = parsed.hostname
    if parsed.port:
        netloc += f':{parsed.port}'
    if parsed.username or parsed.password:
        netloc = f'****:****@{netloc}'
    return f'{parsed.scheme}://{netloc}{parsed.path}'

def validate_csrf_token():
    if request.method not in ('POST', 'PUT', 'PATCH', 'DELETE'):
        return True
    session_token = session.get('csrf_token')
    if not session_token:
        return False
    token = request.form.get('csrf_token') or request.headers.get('X-CSRF-Token')
    if request.is_json:
        payload = request.get_json(silent=True)
        if isinstance(payload, dict):
            token = token or payload.get('csrf_token')
    return bool(token and secrets.compare_digest(str(token), str(session_token)))

@app.before_request
def enforce_security_headers_and_csrf():
    if request.method in ('POST', 'PUT', 'PATCH', 'DELETE') and not validate_csrf_token():
        logger.warning('Potential CSRF attack blocked from %s', request.remote_addr)
        if request.is_json:
            return jsonify({'success': False, 'error': 'Invalid CSRF token'}), 403
        return 'Invalid CSRF token', 403

@app.context_processor
def inject_csrf_token():
    return {'csrf_token': generate_csrf_token(), 'mask_camera_url': mask_camera_url}

def load_cameras(owner_id):
    cameras = {}
    for cam in Camera.query.filter_by(owner_id=owner_id).all():
        cameras[cam.name] = cam.url
    return cameras

def add_camera(owner_id, name, url):
    if Camera.query.filter_by(owner_id=owner_id, name=name).first():
        return False
    new_camera = Camera(owner_id=owner_id, name=name, url=url)
    db.session.add(new_camera)
    db.session.commit()
    return True

def remove_camera_file(owner_id, name):
    cam = Camera.query.filter_by(owner_id=owner_id, name=name).first()
    if cam:
        db.session.delete(cam)
        db.session.commit()
        return True
    return False

def release_camera_connection(camera_url, exclude_camera_id=None):
    """Tear down the pooled connection for camera_url, but only if no other
    Camera row (e.g. a different user pointed at the same physical camera)
    still references that URL - otherwise removing/renaming one user's
    camera would silently kill someone else's live stream."""
    if not camera_url:
        return
    query = Camera.query.filter_by(url=camera_url)
    if exclude_camera_id is not None:
        query = query.filter(Camera.id != exclude_camera_id)
    if query.first():
        return
    with camera_lock:
        _release_camera_locked(camera_url)

@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('login.html')

def is_login_locked(username):
    key = username.lower()
    with login_lock:
        entry = failed_logins.get(key)
        if not entry:
            return False
        count, locked_until = entry
        if locked_until and time.monotonic() < locked_until:
            return True
        if locked_until and time.monotonic() >= locked_until:
            failed_logins.pop(key, None)
        return False

def record_failed_login(username):
    key = username.lower()
    with login_lock:
        count, _ = failed_logins.get(key, (0, None))
        count += 1
        locked_until = time.monotonic() + LOGIN_LOCKOUT_SECONDS if count >= MAX_LOGIN_ATTEMPTS else None
        failed_logins[key] = (count, locked_until)

def clear_failed_logins(username):
    key = username.lower()
    with login_lock:
        failed_logins.pop(key, None)

@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username', '').strip()
    password = request.form.get('password', '').strip()
    if username and is_login_locked(username):
        logger.warning('Login blocked due to lockout for %s', username)
        return render_template('login.html', error='Too many failed attempts. Please try again in a few minutes.')
    user = User.query.filter_by(username=username).first()
    if user and user.check_password(password):
        clear_failed_logins(username)
        login_user(user)
        session.permanent = True
        generate_csrf_token()
        return redirect(url_for('dashboard'))
    if username:
        record_failed_login(username)
    return render_template('login.html', error='Invalid credentials')

@app.route('/dashboard')
@login_required
def dashboard():
    cameras = load_cameras(current_user.id)
    selected_camera = session.get('selected_camera')
    if selected_camera not in cameras:
        session.pop('selected_camera', None)
        selected_camera = None
    return render_template('index.html', cameras=cameras, selected_camera=selected_camera,
                            camera_limit=MAX_CAMERAS_PER_USER)

@app.route('/admin')
@login_required
def admin():
    if not is_admin():
        return redirect(url_for('index'))
    users = User.query.order_by(User.username).all()
    all_cameras = Camera.query.order_by(Camera.owner_id, Camera.name).all()
    return render_template('admin.html', users=users, all_cameras=all_cameras)

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Server Error: {error}")
    return "Internal Server Error. Please check logs.", 500

@app.errorhandler(404)
def not_found_error(error):
    return "Page Not Found", 404

@app.route('/assets/<path:filename>')
def assets(filename):
    if '..' in filename or filename.startswith('/'):
        abort(404)
    return send_from_directory(ASSETS_FOLDER, filename)

@app.after_request
def set_security_headers(response):
    response.headers.setdefault('X-Content-Type-Options', 'nosniff')
    response.headers.setdefault('X-Frame-Options', 'DENY')
    response.headers.setdefault('Referrer-Policy', 'same-origin')
    response.headers.setdefault('Permissions-Policy', 'camera=(), microphone=(), geolocation=()')
    response.headers.setdefault('X-Download-Options', 'noopen')
    response.headers.setdefault('X-XSS-Protection', '1; mode=block')
    response.headers.setdefault(
        'Content-Security-Policy',
        "default-src 'self'; script-src 'self' 'unsafe-inline';"
        " style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self';"
    )
    response.headers.setdefault('Cache-Control', 'no-store')
    response.headers.setdefault('Pragma', 'no-cache')
    response.headers.setdefault('Expires', '0')
    if request.is_secure or os.environ.get('FLASK_ENV', 'production') == 'production':
        response.headers.setdefault('Strict-Transport-Security', 'max-age=31536000; includeSubDomains')
    return response

@app.route('/api/camera/add', methods=['POST'])
@login_required
def api_add_camera():
    data = request.get_json(silent=True) or {}
    camera_name = data.get('camera_name', '').strip()
    camera_url = data.get('camera_url', '').strip()
    if not camera_name or not camera_url:
        return jsonify({'success': False, 'error': 'Camera name and URL are required'}), 400
    if not is_safe_camera_name(camera_name):
        return jsonify({'success': False, 'error': 'Invalid camera name'}), 400
    if not is_safe_camera_url(camera_url):
        return jsonify({'success': False, 'error': 'Invalid or unsupported camera URL'}), 400
    if Camera.query.filter_by(owner_id=current_user.id).count() >= MAX_CAMERAS_PER_USER:
        return jsonify({'success': False, 'error': f'Camera limit reached ({MAX_CAMERAS_PER_USER} max)'}), 400
    if add_camera(current_user.id, camera_name, camera_url):
        logger.info(f"Camera added: {camera_name} (owner={current_user.username})")
        return jsonify({'success': True, 'message': 'Camera added successfully'})
    else:
        return jsonify({'success': False, 'error': 'Camera name already exists'}), 400

@app.route('/api/camera/update', methods=['POST'])
@login_required
def api_update_camera():
    data = request.get_json(silent=True) or {}
    old_name = data.get('old_name', '').strip()
    new_name = data.get('new_name', '').strip()
    new_url = data.get('new_url', '').strip()
    if not old_name or not new_name or not new_url:
        return jsonify({'success': False, 'error': 'All fields are required'}), 400
    if not is_safe_camera_name(old_name) or not is_safe_camera_name(new_name):
        return jsonify({'success': False, 'error': 'Invalid camera name'}), 400
    if not is_safe_camera_url(new_url):
        return jsonify({'success': False, 'error': 'Invalid or unsupported camera URL'}), 400
    cam = Camera.query.filter_by(owner_id=current_user.id, name=old_name).first()
    if not cam:
        return jsonify({'success': False, 'error': 'Camera not found'}), 404
    if old_name != new_name and Camera.query.filter_by(owner_id=current_user.id, name=new_name).first():
        return jsonify({'success': False, 'error': 'Camera name already exists'}), 400
    old_url = cam.url
    cam.name = new_name
    cam.url = new_url
    db.session.commit()
    if session.get('selected_camera') == old_name:
        session['selected_camera'] = new_name
        session.modified = True
    if old_url and old_url != new_url:
        release_camera_connection(old_url, exclude_camera_id=cam.id)
    logger.info(f"Camera updated: {old_name} -> {new_name}")
    return jsonify({'success': True, 'message': 'Camera updated successfully'})

@app.route('/api/camera/remove', methods=['POST'])
@login_required
def api_remove_camera():
    data = request.get_json(silent=True) or {}
    camera_name = data.get('camera_name', '').strip()
    if not camera_name or not is_safe_camera_name(camera_name):
        return jsonify({'success': False, 'error': 'Camera name is required and must be valid'}), 400
    cameras = load_cameras(current_user.id)
    camera_url = cameras.get(camera_name)
    if remove_camera_file(current_user.id, camera_name):
        if session.get('selected_camera') == camera_name:
            session['selected_camera'] = None
            session.modified = True
        if camera_url:
            release_camera_connection(camera_url)
        logger.info(f"Camera removed: {camera_name}")
        return jsonify({'success': True, 'message': 'Camera removed successfully'})
    else:
        return jsonify({'success': False, 'error': 'Camera not found'}), 404

@app.route('/camera_status')
@login_required
def camera_status():
    cameras = load_cameras(current_user.id)
    selected = session.get('selected_camera')
    camera_url = cameras.get(selected, '')
    if not camera_url or not is_safe_camera_url(camera_url):
        return jsonify({'status': 'error', 'message': 'No valid camera selected'})
    # Reuse the same pooled/timeout-hardened connection path as the live
    # stream, so "Test Connection" reflects reality and pre-warms the
    # connection for the stream that's about to follow.
    cap = get_camera(camera_url)
    if not cap:
        return jsonify({'status': 'error', 'message': 'Failed to connect to camera'})
    cap.grab()
    ret, frame = cap.retrieve()
    if ret and frame is not None:
        return jsonify({'status': 'success', 'message': 'Camera connected'})
    return jsonify({'status': 'error', 'message': 'Camera connected but no frame received'})

@app.route('/select_camera', methods=['POST'])
@login_required
def select_camera():
    selected_camera = request.form.get('selected_camera', '').strip()
    logger.info(f"Camera selection request: {selected_camera}")
    cameras = load_cameras(current_user.id)
    if selected_camera and selected_camera in cameras:
        session['selected_camera'] = selected_camera
        session.modified = True
        logger.info(f"Camera selected successfully: {selected_camera}")
    else:
        logger.warning(f"Invalid camera selection: {selected_camera}")
    return redirect(url_for('dashboard'))

@app.route('/healthz')
def healthz():
    try:
        db.session.execute(db.text('SELECT 1'))
        return jsonify({'status': 'ok'})
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 503

@app.route('/golive')
@login_required
def video_feed():
    cameras = load_cameras(current_user.id)
    camera_url = cameras.get(session.get('selected_camera', ''), '')
    if not camera_url or not is_safe_camera_url(camera_url):
        return "No camera selected or configured", 400
    return Response(stream_frames(camera_url, apply_ai=False),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/goai')
@login_required
def video_feedai():
    cameras = load_cameras(current_user.id)
    camera_url = cameras.get(session.get('selected_camera', ''), '')
    if not camera_url or not is_safe_camera_url(camera_url):
        return "No camera selected or configured", 400
    return Response(stream_frames(camera_url, apply_ai=True),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed/<camera_name>')
@login_required
def video_feed_named(camera_name):
    cameras = load_cameras(current_user.id)
    camera_url = cameras.get(camera_name)
    if not camera_url or not is_safe_camera_url(camera_url):
        return f"Camera '{camera_name}' not found", 404
    return Response(stream_frames(camera_url, apply_ai=False),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_ai/<camera_name>')
@login_required
def video_feed_ai_named(camera_name):
    cameras = load_cameras(current_user.id)
    camera_url = cameras.get(camera_name)
    if not camera_url or not is_safe_camera_url(camera_url):
        return f"Camera '{camera_name}' not found", 404
    return Response(stream_frames(camera_url, apply_ai=True),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def doAI(img):
    if img is None:
        logger.warning("AI processing received empty frame")
        return None
    if model is None:
        logger.warning("YOLO model unavailable, skipping AI inference")
        return cv2.resize(img, (640, 480))
    try:
        img_resized = cv2.resize(img, (640, 480))
        results = model(img_resized, verbose=False)[0]
        names = results.names
        boxes = results.boxes
        person, vehicles = 0, 0
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = box.conf[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = names[cls_id]
            if conf > 0.6:
                cv2.rectangle(img_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_resized, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                if label == 'person':
                    person += 1
                elif label in ['car', 'bus', 'truck']:
                    vehicles += 1
        cv2.putText(img_resized, f'Person: {person}, Vehicles: {vehicles}', (10, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 2)
        return img_resized
    except Exception as e:
        logger.error(f"Error in AI processing: {e}")
        return cv2.resize(img, (640, 480))

def _webcam_backend():
    system = platform.system()
    if system == 'Windows':
        return cv2.CAP_DSHOW
    if system == 'Darwin':
        return cv2.CAP_AVFOUNDATION
    if system == 'Linux':
        return cv2.CAP_V4L2
    return cv2.CAP_ANY

def _open_capture(camera_url):
    """Actually connect to a camera. Slow and blocking (this is the whole
    point of keeping it out from under camera_lock) - callers must not hold
    camera_lock while calling this."""
    cap = None
    if camera_url.startswith('webcam:'):
        camera_index = int(camera_url.split(':', 1)[1])
        preferred = _webcam_backend()
        cap = cv2.VideoCapture(camera_index, preferred)
        if not cap.isOpened():
            if cap:
                cap.release()
            logger.warning(f"Preferred backend {preferred} failed for webcam:{camera_index}, trying default")
            cap = cv2.VideoCapture(camera_index, cv2.CAP_ANY)
    else:
        # Set open/read timeouts *before* connecting - setting them on an
        # already-open (or already hung) capture is too late, since the
        # blocking connect attempt happens inside the VideoCapture() call
        # itself.
        timeout_params = [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, CAMERA_CONNECT_TIMEOUT_MS,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC, CAMERA_READ_TIMEOUT_MS,
        ]
        for backend in (cv2.CAP_FFMPEG, cv2.CAP_ANY):
            try:
                logger.info(f"Trying backend {backend} for {camera_url}")
                cap = cv2.VideoCapture(camera_url, backend, timeout_params)
                if cap.isOpened():
                    logger.info(f"Successfully opened with backend {backend}")
                    break
            except Exception as e:
                logger.error(f"Backend {backend} failed: {e}")
                cap = None
            if cap and not cap.isOpened():
                cap.release()
                cap = None

    if cap and cap.isOpened():
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # Smaller buffer for lower latency
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return cap

    if cap:
        cap.release()
    return None

def _release_camera_locked(camera_url):
    """Release and forget a camera. Caller must hold camera_lock."""
    cap = active_cameras.pop(camera_url, None)
    camera_last_used.pop(camera_url, None)
    if cap:
        try:
            cap.release()
        except Exception:
            pass

def _evict_lru_camera_locked():
    """Free a slot by releasing the least-recently-used active camera, but
    only if it's been idle long enough that we're not likely to be yanking
    a connection out from under someone actively watching it. Caller must
    hold camera_lock. Returns True if a slot was freed."""
    if not camera_last_used:
        return False
    lru_url, last_used = min(camera_last_used.items(), key=lambda kv: kv[1])
    if time.monotonic() - last_used < CAMERA_EVICTION_GRACE_SECONDS:
        return False
    logger.info(f"Evicting least-recently-used camera to free a slot: {lru_url}")
    _release_camera_locked(lru_url)
    return True

def get_camera(camera_url):
    if not is_safe_camera_url(camera_url):
        logger.warning("Rejected unsafe camera URL: %s", camera_url)
        return None

    with camera_lock:
        cap = active_cameras.get(camera_url)
        if cap:
            if cap.isOpened():
                camera_last_used[camera_url] = time.monotonic()
                return cap
            _release_camera_locked(camera_url)
        # One lock per URL: concurrent requests for *this* camera serialize
        # below (so we never open duplicate connections to the same
        # camera), but requests for *other* cameras are unaffected.
        url_lock = camera_url_locks.setdefault(camera_url, threading.Lock())

    with url_lock:
        with camera_lock:
            # Another thread may have already connected while we waited.
            cap = active_cameras.get(camera_url)
            if cap and cap.isOpened():
                camera_last_used[camera_url] = time.monotonic()
                return cap
            if len(active_cameras) >= MAX_ACTIVE_CAMERAS and not _evict_lru_camera_locked():
                logger.warning(f"Max active cameras limit ({MAX_ACTIVE_CAMERAS}) reached")
                return None

        logger.info(f"Creating new capture for {camera_url}")
        cap = _open_capture(camera_url)  # slow + blocking; camera_lock NOT held here

        with camera_lock:
            if cap:
                active_cameras[camera_url] = cap
                camera_last_used[camera_url] = time.monotonic()
                logger.info(f"Camera stream opened successfully: {camera_url}")
                return cap
            logger.error(f"Failed to open camera stream: {camera_url}")
            return None

def _reap_idle_cameras():
    while True:
        time.sleep(30)
        try:
            with camera_lock:
                now = time.monotonic()
                idle_urls = [
                    url for url, last_used in camera_last_used.items()
                    if now - last_used > CAMERA_IDLE_TIMEOUT_SECONDS
                ]
                for url in idle_urls:
                    logger.info(f"Releasing idle camera connection: {url}")
                    _release_camera_locked(url)
        except Exception as e:
            logger.error(f"Error in camera idle reaper: {e}")

def stream_frames(camera_url, apply_ai=False):
    cap = None
    consecutive_failures = 0
    max_failures = 50

    try:
        while True:
            if not cap or not cap.isOpened():
                logger.info(f"Reconnecting to camera: {camera_url}")
                cap = get_camera(camera_url)
                consecutive_failures = 0
                if not cap:
                    logger.error(f"Failed to get camera for reconnection: {camera_url}")
                    time.sleep(2)
                    continue

            # Flush any buffered/stale frames cheaply: grab() only demuxes,
            # it doesn't decode, unlike read() which does a full decode.
            cap.grab()
            cap.grab()

            ret, frame = cap.retrieve()
            if not ret or frame is None:
                consecutive_failures += 1
                logger.warning(f"Failed to read frame (failures: {consecutive_failures}/{max_failures})")
                if consecutive_failures >= max_failures:
                    logger.error(f"Too many failures, releasing camera: {camera_url}")
                    with camera_lock:
                        _release_camera_locked(camera_url)
                    cap = None
                time.sleep(0.1)
                continue

            consecutive_failures = 0
            with camera_lock:
                if camera_url in camera_last_used:
                    camera_last_used[camera_url] = time.monotonic()

            if apply_ai:
                frame = doAI(frame)
            else:
                frame = cv2.resize(frame, (480, 360))

            if frame is None:
                time.sleep(0.001)
                continue

            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
            _, buffer = cv2.imencode('.jpg', frame, encode_param)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    except Exception as e:
        logger.error(f"Error in stream_frames for {camera_url}: {e}")
        if cap:
            with camera_lock:
                # Only tear down the pooled connection if it's still this
                # same object - another request may have already reconnected.
                if active_cameras.get(camera_url) is cap:
                    _release_camera_locked(camera_url)

@app.route('/logout', methods=['POST'])
@login_required
def logout():
    if not validate_csrf_token():
        return redirect(url_for('index'))
    # Deliberately does NOT touch active_cameras: those connections are a
    # shared pool across all users (keyed by URL, not by session), and other
    # people may currently be streaming from them. Idle ones are cleaned up
    # by the background reaper instead.
    logout_user()
    session.clear()
    return redirect(url_for('index'))

@app.route('/api/user/add', methods=['POST'])
@login_required
def api_add_user():
    if not is_admin():
        return jsonify({'success': False, 'error': 'Not authorized'}), 403
    data = request.get_json(silent=True) or {}
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()
    role = data.get('role', 'user').strip()
    if not username or not password:
        return jsonify({'success': False, 'error': 'Username and password are required'}), 400
    if not is_valid_username(username):
        return jsonify({'success': False, 'error': 'Username must be 3-32 characters (letters, numbers, _ . -)'}), 400
    if not is_valid_password(password):
        return jsonify({'success': False, 'error': 'Password must be at least 8 characters'}), 400
    if User.query.filter_by(username=username).first():
        return jsonify({'success': False, 'error': 'Username already exists'}), 400
    if role not in ['user', 'admin']:
        return jsonify({'success': False, 'error': 'Invalid role'}), 400
    new_user = User(username=username, role=role)
    new_user.set_password(password)
    db.session.add(new_user)
    db.session.commit()
    logger.info(f"User added: {username}")
    return jsonify({'success': True, 'message': 'User added successfully'})

@app.route('/api/user/update', methods=['POST'])
@login_required
def api_update_user():
    if not is_admin():
        return jsonify({'success': False, 'error': 'Not authorized'}), 403
    data = request.get_json(silent=True) or {}
    try:
        user_id = int(data.get('user_id'))
    except (TypeError, ValueError):
        return jsonify({'success': False, 'error': 'Invalid user ID'}), 400
    user = User.query.get(user_id)
    if not user:
        return jsonify({'success': False, 'error': 'User not found'}), 404

    new_username = data.get('username', '').strip()
    new_password = data.get('password', '').strip()
    new_role = data.get('role', '').strip()

    if new_username and new_username != user.username:
        if not is_valid_username(new_username):
            return jsonify({'success': False, 'error': 'Username must be 3-32 characters (letters, numbers, _ . -)'}), 400
        if User.query.filter(User.username == new_username, User.id != user.id).first():
            return jsonify({'success': False, 'error': 'Username already exists'}), 400
        user.username = new_username

    if new_password:
        if not is_valid_password(new_password):
            return jsonify({'success': False, 'error': 'Password must be at least 8 characters'}), 400
        user.set_password(new_password)

    if new_role and new_role != user.role:
        if new_role not in ['user', 'admin']:
            return jsonify({'success': False, 'error': 'Invalid role'}), 400
        if user.role == 'admin' and new_role != 'admin' and admin_count(exclude_user_id=user.id) < 1:
            return jsonify({'success': False, 'error': 'Cannot demote the last remaining admin'}), 400
        user.role = new_role

    db.session.commit()
    logger.info(f"User updated: {user.username} (id={user.id})")
    return jsonify({'success': True, 'message': 'User updated successfully'})

@app.route('/api/user/remove', methods=['POST'])
@login_required
def api_remove_user():
    if not is_admin():
        return jsonify({'success': False, 'error': 'Not authorized'}), 403
    data = request.get_json(silent=True) or {}
    try:
        user_id = int(data.get('user_id'))
    except (TypeError, ValueError):
        return jsonify({'success': False, 'error': 'Invalid user ID'}), 400
    if user_id == current_user.id:
        return jsonify({'success': False, 'error': 'Cannot delete your own account'}), 400
    user = User.query.get(user_id)
    if not user:
        return jsonify({'success': False, 'error': 'User not found'}), 404
    if user.role == 'admin' and admin_count(exclude_user_id=user.id) < 1:
        return jsonify({'success': False, 'error': 'Cannot delete the last remaining admin'}), 400
    db.session.delete(user)
    db.session.commit()
    logger.info(f"User removed: {user.username}")
    return jsonify({'success': True, 'message': 'User removed successfully'})

@app.route('/api/user/change_password', methods=['POST'])
@login_required
def api_change_password():
    data = request.get_json(silent=True) or {}
    current_password = data.get('current_password', '').strip()
    new_password = data.get('new_password', '').strip()
    if not current_password or not new_password:
        return jsonify({'success': False, 'error': 'Current and new password are required'}), 400
    if not current_user.check_password(current_password):
        return jsonify({'success': False, 'error': 'Current password is incorrect'}), 400
    if not is_valid_password(new_password):
        return jsonify({'success': False, 'error': 'New password must be at least 8 characters'}), 400
    current_user.set_password(new_password)
    db.session.commit()
    logger.info(f"Password changed for user: {current_user.username}")
    return jsonify({'success': True, 'message': 'Password changed successfully'})

@app.route('/api/admin/camera/remove', methods=['POST'])
@login_required
def api_admin_remove_camera():
    if not is_admin():
        return jsonify({'success': False, 'error': 'Not authorized'}), 403
    data = request.get_json(silent=True) or {}
    try:
        camera_id = int(data.get('camera_id'))
    except (TypeError, ValueError):
        return jsonify({'success': False, 'error': 'Invalid camera ID'}), 400
    cam = Camera.query.get(camera_id)
    if not cam:
        return jsonify({'success': False, 'error': 'Camera not found'}), 404
    camera_url = cam.url
    logger.info(f"Admin removing camera: {cam.name} (owner_id={cam.owner_id})")
    db.session.delete(cam)
    db.session.commit()
    if camera_url:
        release_camera_connection(camera_url)
    return jsonify({'success': True, 'message': 'Camera removed successfully'})

if __name__ == '__main__':
    threading.Thread(target=_reap_idle_cameras, daemon=True).start()
    # threaded=True is load-bearing: /golive, /goai and friends are
    # long-lived MJPEG generators that never return, so a non-threaded dev
    # server would serve exactly one such connection at a time and lock out
    # every other user (including other requests from the same user) for as
    # long as that stream stays open. For real deployments, use a proper
    # multi-worker/multi-thread WSGI server (see deployment docs) instead of
    # this dev server.
    app.run(
        debug=False,
        host=os.environ.get('FLASK_RUN_HOST', '0.0.0.0'),
        port=int(os.environ.get('FLASK_RUN_PORT', 8000)),
        threaded=True,
    )
