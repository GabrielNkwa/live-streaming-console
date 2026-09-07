from flask import Flask, render_template, request, Response, session, redirect, url_for, jsonify, abort, send_from_directory
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from sqlalchemy import inspect, text
import cv2
from ultralytics import YOLO
import threading
import time
import logging
import ipaddress
import os
import secrets
import re
from functools import wraps
from datetime import timedelta
from urllib.parse import urlparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("Starting Flask app initialization")

app = Flask(__name__, static_url_path=None)

secret_key = os.environ.get('FLASK_SECRET_KEY')
if not secret_key:
    secret_key = secrets.token_urlsafe(32)
    logger.warning("Using fallback ephemeral secret key. Set FLASK_SECRET_KEY in production.")
app.secret_key = secret_key

app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    SESSION_COOKIE_SECURE=os.environ.get('FLASK_ENV', 'production') != 'development',
    PERMANENT_SESSION_LIFETIME=timedelta(minutes=20),
    PREFERRED_URL_SCHEME='https',
    SQLALCHEMY_DATABASE_URI=os.environ.get('DATABASE_URL', 'sqlite:///dlc.db'),
    SQLALCHEMY_TRACK_MODIFICATIONS=False,
    # Mutation endpoints only ever receive a handful of short fields (names, URLs,
    # credentials) as JSON - cap the body size so a client can't tie up a worker
    # thread parsing/buffering an oversized request.
    MAX_CONTENT_LENGTH=256 * 1024,
)

if os.environ.get('USE_PROXY_FIX') == '1':
    app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

ASSETS_FOLDER = os.path.join(app.root_path, 'static', 'assets')
os.makedirs(ASSETS_FOLDER, exist_ok=True)

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'index'

limiter = Limiter(
    get_remote_address,
    app=app,
    storage_uri="memory://",
    default_limits=[],
)

logger.info("Loading YOLO model")
try:
    model = YOLO('yolo11n.pt')
    logger.info("YOLO model loaded successfully")
except Exception as e:
    logger.error(f"Error loading YOLO model: {e}")
    model = None

MAX_ACTIVE_CAMERAS = 10


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(20), nullable=False, default='user')

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class Camera(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64), nullable=False)
    url = db.Column(db.String(200), nullable=False)
    owner_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    owner = db.relationship('User', backref=db.backref('cameras', lazy=True, cascade='all, delete-orphan'))

    __table_args__ = (
        db.UniqueConstraint('owner_id', 'name', name='uq_camera_owner_name'),
    )


def _migrate_camera_table_to_per_user():
    """Cameras used to be a single global list (name unique system-wide). Moving to
    per-user ownership needs a real schema change - SQLite can't ADD a composite
    UNIQUE constraint or drop the old column-level one via ALTER TABLE, so an
    existing table is rebuilt: renamed aside, recreated with the new schema, and
    its rows copied over with every camera assigned to the first admin account
    (the only reasonable owner for cameras that pre-date per-user cameras)."""
    inspector = inspect(db.engine)
    if 'camera' not in inspector.get_table_names():
        return  # fresh DB - db.create_all() right after this will make the new schema
    columns = {col['name'] for col in inspector.get_columns('camera')}
    if 'owner_id' in columns:
        return  # already migrated
    logger.info("Migrating camera table to per-user ownership")
    with db.engine.begin() as conn:
        conn.execute(text('ALTER TABLE camera RENAME TO camera_legacy'))
        conn.execute(text('''
            CREATE TABLE camera (
                id INTEGER NOT NULL PRIMARY KEY,
                name VARCHAR(64) NOT NULL,
                url VARCHAR(200) NOT NULL,
                owner_id INTEGER NOT NULL,
                FOREIGN KEY(owner_id) REFERENCES user (id),
                CONSTRAINT uq_camera_owner_name UNIQUE (owner_id, name)
            )
        '''))
        first_admin = conn.execute(
            text("SELECT id FROM user WHERE role = 'admin' ORDER BY id LIMIT 1")
        ).fetchone()
        if first_admin:
            conn.execute(
                text('INSERT INTO camera (id, name, url, owner_id) '
                     'SELECT id, name, url, :oid FROM camera_legacy'),
                {'oid': first_admin[0]}
            )
            logger.info(f"Assigned {conn.execute(text('SELECT COUNT(*) FROM camera')).scalar()} "
                        f"pre-existing camera(s) to admin id={first_admin[0]}")
        else:
            logger.warning("No admin user found - pre-existing cameras were not migrated")
        conn.execute(text('DROP TABLE camera_legacy'))
    logger.info("Camera table migration complete")


# Ensure the schema exists even if init_db.py was never run, so a fresh
# checkout doesn't 500 on first request. Idempotent - only creates missing tables.
with app.app_context():
    _migrate_camera_table_to_per_user()
    db.create_all()
    try:
        weak_admin = User.query.filter_by(role='admin').first()
        if weak_admin and weak_admin.check_password('admin'):
            logger.warning(
                "SECURITY WARNING: admin account '%s' is still using the default password. "
                "Change it from the admin panel immediately.", weak_admin.username
            )
    except Exception as e:
        logger.debug("Skipped default-password check: %s", e)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


def generate_csrf_token():
    token = session.get('csrf_token')
    if not token:
        token = secrets.token_urlsafe(32)
        session['csrf_token'] = token
    return token


CAMERA_NAME_RE = re.compile(r"[A-Za-z0-9 _\-]{1,64}")
USERNAME_RE = re.compile(r"[A-Za-z0-9_.\-]{3,32}")


def is_safe_camera_name(camera_name):
    return bool(re.fullmatch(CAMERA_NAME_RE, camera_name))


def is_safe_username(username):
    return bool(re.fullmatch(USERNAME_RE, username))


def _is_blocked_host(hostname):
    """Blocks loopback/link-local/metadata-style addresses so an admin can't point
    a camera at the server itself or cloud metadata endpoints. RFC1918 LAN ranges
    (192.168.x, 10.x, 172.16-31.x) are intentionally left open - that's where the
    actual cameras live."""
    hostname = hostname.lower().strip('[]')
    if hostname == 'localhost':
        return True
    try:
        ip = ipaddress.ip_address(hostname)
    except ValueError:
        return False
    return ip.is_loopback or ip.is_link_local or ip.is_unspecified or ip.is_multicast or ip.is_reserved


def is_safe_camera_url(camera_url):
    if not camera_url or len(camera_url) > 200:
        return False
    if camera_url.startswith('webcam:'):
        try:
            index = int(camera_url.split(':', 1)[1])
            return 0 <= index <= 4
        except ValueError:
            return False
    parsed = urlparse(camera_url)
    if parsed.scheme not in ('rtsp', 'http', 'https'):
        return False
    if not parsed.hostname:
        return False
    if _is_blocked_host(parsed.hostname):
        return False
    return True


def is_admin():
    return current_user.is_authenticated and current_user.role == 'admin'


def admin_required(view):
    @wraps(view)
    @login_required
    def wrapped(*args, **kwargs):
        if not is_admin():
            if request.path.startswith('/api/'):
                return jsonify({'success': False, 'error': 'Admin privileges required'}), 403
            return redirect(url_for('dashboard'))
        return view(*args, **kwargs)
    return wrapped


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
    return {'csrf_token': generate_csrf_token()}


def load_cameras_for_user(user_id):
    """A user's own cameras only - the dashboard, camera selection, and video
    streaming are all scoped to this so one user can never see or stream
    another user's camera, admins included (their oversight is on /admin)."""
    cameras = {}
    for cam in Camera.query.filter_by(owner_id=user_id).all():
        cameras[cam.name] = cam.url
    return cameras


def add_camera(name, url, owner_id):
    if Camera.query.filter_by(owner_id=owner_id, name=name).first():
        return False
    new_camera = Camera(name=name, url=url, owner_id=owner_id)
    db.session.add(new_camera)
    db.session.commit()
    return True


def remove_camera_file(name, owner_id):
    cam = Camera.query.filter_by(owner_id=owner_id, name=name).first()
    if cam:
        db.session.delete(cam)
        db.session.commit()
        return True
    return False


@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('login.html')


@app.route('/login', methods=['POST'])
@limiter.limit("8 per minute")
def login():
    username = request.form.get('username', '').strip()
    password = request.form.get('password', '').strip()
    user = User.query.filter_by(username=username).first()
    if user and user.check_password(password):
        login_user(user)
        session.permanent = True
        generate_csrf_token()
        return redirect(url_for('dashboard'))
    logger.warning('Failed login attempt for username=%r from %s', username, request.remote_addr)
    return render_template('login.html', error='Invalid credentials')


@app.route('/dashboard')
@login_required
def dashboard():
    cameras = load_cameras_for_user(current_user.id)
    selected_camera = session.get('selected_camera')
    if selected_camera not in cameras:
        session.pop('selected_camera', None)
        selected_camera = None
    return render_template('index.html', cameras=cameras, selected_camera=selected_camera)


@app.route('/admin')
@admin_required
def admin():
    # Oversight view: every camera across every user, not just the admin's own.
    cameras = sorted(Camera.query.all(), key=lambda c: (c.owner.username.lower(), c.name.lower()))
    users = User.query.order_by(User.username).all()
    return render_template('admin.html', cameras=cameras, users=users)


@app.errorhandler(429)
def ratelimit_handler(e):
    logger.warning('Rate limit hit for %s on %s', request.remote_addr, request.path)
    if request.is_json or request.path.startswith('/api/'):
        return jsonify({'success': False, 'error': 'Too many requests, please slow down'}), 429
    return render_template('login.html', error='Too many attempts. Please wait a moment and try again.'), 429


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
@limiter.limit("20 per minute")
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
    if add_camera(camera_name, camera_url, current_user.id):
        logger.info(f"Camera added by {current_user.username}: {camera_name}")
        return jsonify({'success': True, 'message': 'Camera added successfully'})
    else:
        return jsonify({'success': False, 'error': 'You already have a camera with that name'}), 400


@app.route('/api/camera/update', methods=['POST'])
@login_required
@limiter.limit("20 per minute")
def api_update_camera():
    """Self-service: a user can only ever edit their own camera. Admin oversight
    edits of any user's camera go through /api/admin/camera/update instead."""
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
        return jsonify({'success': False, 'error': 'You already have a camera with that name'}), 400
    old_url = cam.url
    cam.name = new_name
    cam.url = new_url
    db.session.commit()
    if session.get('selected_camera') == old_name:
        session['selected_camera'] = new_name
        session.modified = True
    if old_url and old_url != new_url:
        stop_stream(old_url)
    logger.info(f"Camera updated by {current_user.username}: {old_name} -> {new_name}")
    return jsonify({'success': True, 'message': 'Camera updated successfully'})


@app.route('/api/camera/remove', methods=['POST'])
@login_required
@limiter.limit("20 per minute")
def api_remove_camera():
    """Self-service removal of one of the current user's own cameras."""
    data = request.get_json(silent=True) or {}
    camera_name = data.get('camera_name', '').strip()
    if not camera_name or not is_safe_camera_name(camera_name):
        return jsonify({'success': False, 'error': 'Camera name is required and must be valid'}), 400
    cameras = load_cameras_for_user(current_user.id)
    camera_url = cameras.get(camera_name)
    if remove_camera_file(camera_name, current_user.id):
        if session.get('selected_camera') == camera_name:
            session['selected_camera'] = None
            session.modified = True
        if camera_url:
            stop_stream(camera_url)
        logger.info(f"Camera removed by {current_user.username}: {camera_name}")
        return jsonify({'success': True, 'message': 'Camera removed successfully'})
    else:
        return jsonify({'success': False, 'error': 'Camera not found'}), 404


@app.route('/api/admin/camera/update', methods=['POST'])
@admin_required
@limiter.limit("20 per minute")
def api_admin_update_camera():
    """Oversight edit of any user's camera, keyed by id since names are only
    unique per-owner now."""
    data = request.get_json(silent=True) or {}
    try:
        camera_id = int(data.get('camera_id'))
    except (TypeError, ValueError):
        return jsonify({'success': False, 'error': 'Invalid camera ID'}), 400
    new_name = data.get('new_name', '').strip()
    new_url = data.get('new_url', '').strip()
    if not new_name or not new_url:
        return jsonify({'success': False, 'error': 'All fields are required'}), 400
    if not is_safe_camera_name(new_name):
        return jsonify({'success': False, 'error': 'Invalid camera name'}), 400
    if not is_safe_camera_url(new_url):
        return jsonify({'success': False, 'error': 'Invalid or unsupported camera URL'}), 400
    cam = Camera.query.get(camera_id)
    if not cam:
        return jsonify({'success': False, 'error': 'Camera not found'}), 404
    old_name, old_url = cam.name, cam.url
    if new_name != old_name and Camera.query.filter_by(owner_id=cam.owner_id, name=new_name).first():
        return jsonify({'success': False, 'error': 'That owner already has a camera with that name'}), 400
    cam.name = new_name
    cam.url = new_url
    db.session.commit()
    if old_url and old_url != new_url:
        stop_stream(old_url)
    logger.info(f"Camera #{camera_id} updated by admin {current_user.username}: {old_name} -> {new_name}")
    return jsonify({'success': True, 'message': 'Camera updated successfully'})


@app.route('/api/admin/camera/remove', methods=['POST'])
@admin_required
@limiter.limit("20 per minute")
def api_admin_remove_camera():
    """Oversight removal of any user's camera, keyed by id."""
    data = request.get_json(silent=True) or {}
    try:
        camera_id = int(data.get('camera_id'))
    except (TypeError, ValueError):
        return jsonify({'success': False, 'error': 'Invalid camera ID'}), 400
    cam = Camera.query.get(camera_id)
    if not cam:
        return jsonify({'success': False, 'error': 'Camera not found'}), 404
    camera_name, camera_url, owner_id = cam.name, cam.url, cam.owner_id
    db.session.delete(cam)
    db.session.commit()
    # The owner's session (if any) self-heals on next /dashboard load - it drops
    # a selected_camera that no longer resolves to one of their cameras.
    if camera_url:
        stop_stream(camera_url)
    logger.info(f"Camera #{camera_id} ({camera_name}) removed by admin {current_user.username}, was owned by user #{owner_id}")
    return jsonify({'success': True, 'message': 'Camera removed successfully'})


@app.route('/camera_status')
@login_required
def camera_status():
    cameras = load_cameras_for_user(current_user.id)
    selected = session.get('selected_camera')
    camera_url = cameras.get(selected, '')
    if not camera_url or not is_safe_camera_url(camera_url):
        return jsonify({'status': 'error', 'message': 'No valid camera selected'})

    # If the shared stream for this camera is already running, trust it instead
    # of opening a second capture on the same device - many webcams (and some
    # RTSP servers) can't serve two concurrent connections.
    with camera_registry_lock:
        stream = camera_registry.get(camera_url)
    if stream is not None and stream.is_alive():
        if stream.has_frame():
            return jsonify({'status': 'success', 'message': 'Camera connected'})
        return jsonify({'status': 'error', 'message': 'Camera connecting, no frame yet'})

    if camera_url.startswith('webcam:'):
        camera_index = int(camera_url.split(':', 1)[1])
        cap = cv2.VideoCapture(camera_index)
    else:
        cap = cv2.VideoCapture(camera_url)
    try:
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                return jsonify({'status': 'success', 'message': 'Camera connected'})
            return jsonify({'status': 'error', 'message': 'Camera connected but no frame received'})
        return jsonify({'status': 'error', 'message': 'Failed to connect to camera'})
    finally:
        cap.release()


@app.route('/select_camera', methods=['POST'])
@login_required
def select_camera():
    selected_camera = request.form.get('selected_camera', '').strip()
    logger.info(f"Camera selection request: {selected_camera}")
    cameras = load_cameras_for_user(current_user.id)
    if selected_camera and selected_camera in cameras:
        session['selected_camera'] = selected_camera
        session.modified = True
        logger.info(f"Camera selected successfully: {selected_camera}")
    else:
        logger.warning(f"Invalid camera selection: {selected_camera}")
    return redirect(url_for('dashboard'))


@app.route('/test')
def test():
    return "Test route works!"


logger.info("Test route registered")


@app.route('/golive')
@login_required
def video_feed():
    cameras = load_cameras_for_user(current_user.id)
    camera_url = cameras.get(session.get('selected_camera', ''), '')
    if not camera_url or not is_safe_camera_url(camera_url):
        return "No camera selected or configured", 400
    return Response(mjpeg_generator(camera_url, apply_ai=False),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/goai')
@login_required
def video_feedai():
    cameras = load_cameras_for_user(current_user.id)
    camera_url = cameras.get(session.get('selected_camera', ''), '')
    if not camera_url or not is_safe_camera_url(camera_url):
        return "No camera selected or configured", 400
    return Response(mjpeg_generator(camera_url, apply_ai=True),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed/<camera_name>')
@login_required
def video_feed_named(camera_name):
    cameras = load_cameras_for_user(current_user.id)
    camera_url = cameras.get(camera_name)
    if not camera_url or not is_safe_camera_url(camera_url):
        return f"Camera '{camera_name}' not found", 404
    return Response(mjpeg_generator(camera_url, apply_ai=False),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed_ai/<camera_name>')
@login_required
def video_feed_ai_named(camera_name):
    cameras = load_cameras_for_user(current_user.id)
    camera_url = cameras.get(camera_name)
    if not camera_url or not is_safe_camera_url(camera_url):
        return f"Camera '{camera_name}' not found", 404
    return Response(mjpeg_generator(camera_url, apply_ai=True),
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


class CameraStream:
    """Owns a single cv2.VideoCapture for one camera URL and reads it from one
    background thread. All HTTP viewers (raw or AI, single-camera or grid view)
    read the latest already-decoded frame from here instead of touching the
    capture themselves - this is what used to happen (every open MJPEG request
    ran its own read loop against the *same* VideoCapture with no locking),
    which is both a thread-safety hazard and wasteful when several people watch
    the same camera. AI inference is throttled to a fraction of frames since
    YOLO is far more expensive than a JPEG re-encode."""

    TARGET_FPS = 15
    AI_EVERY_N_FRAMES = 3
    MAX_CONSECUTIVE_FAILURES = 50
    IDLE_TIMEOUT_SECONDS = 20

    def __init__(self, camera_url):
        self.camera_url = camera_url
        self.cap = None
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.latest_jpeg = None
        self.latest_ai_jpeg = None
        self.frame_version = 0
        self.ai_frame_version = 0
        self.viewers = 0
        self.last_viewer_at = time.monotonic()
        self.running = False
        self.thread = None
        self._frame_counter = 0

    def start(self):
        with self.lock:
            if self.running:
                return
            self.running = True
        self.thread = threading.Thread(target=self._run, name=f"camstream:{self.camera_url}", daemon=True)
        self.thread.start()

    def stop(self):
        with self.lock:
            self.running = False
            self.condition.notify_all()
        if self.thread:
            self.thread.join(timeout=3)

    def is_alive(self):
        with self.lock:
            return self.running

    def has_frame(self):
        with self.lock:
            return self.latest_jpeg is not None

    def add_viewer(self):
        with self.lock:
            self.viewers += 1
            self.last_viewer_at = time.monotonic()

    def remove_viewer(self):
        with self.lock:
            self.viewers = max(0, self.viewers - 1)
            self.last_viewer_at = time.monotonic()

    def get_frame(self, ai, last_version, timeout=15):
        with self.condition:
            deadline = time.monotonic() + timeout
            while True:
                version = self.ai_frame_version if ai else self.frame_version
                jpeg = self.latest_ai_jpeg if ai else self.latest_jpeg
                if jpeg is not None and version != last_version:
                    return jpeg, version
                if not self.running:
                    return None, version
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None, version
                self.condition.wait(timeout=remaining)

    def _open_capture(self):
        camera_url = self.camera_url
        cap = None

        if camera_url.startswith('webcam:'):
            camera_index = int(camera_url.split(':', 1)[1])
            cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                logger.warning("CAP_DSHOW failed, trying default backend for webcam")
                cap = cv2.VideoCapture(camera_index)
        else:
            # The open/read timeouts must be passed into the VideoCapture constructor,
            # not set on the object afterwards - VideoCapture() connects synchronously,
            # so by the time isOpened() can be checked it has already blocked for as
            # long as the OS/codec takes to give up. Measured against an unreachable
            # host: CAP_FFMPEG honors these and fails in ~3-5s; CAP_ANY ignored them
            # entirely and stayed wedged for 90+ minutes with no way to cancel it, so
            # it is deliberately not used here - CAP_FFMPEG alone covers rtsp/http/https,
            # the only schemes is_safe_camera_url allows.
            open_params = [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000,
            ]
            try:
                logger.info(f"Opening {camera_url} via CAP_FFMPEG")
                cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG, open_params)
            except Exception as e:
                logger.error(f"CAP_FFMPEG failed for {camera_url}: {e}")
                if cap:
                    try:
                        cap.release()
                    except Exception:
                        pass
                cap = None

        if cap and cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            logger.info(f"Camera stream opened successfully: {camera_url}")
            return cap

        if cap:
            try:
                cap.release()
            except Exception:
                pass
        logger.error(f"Failed to open camera stream: {camera_url}")
        return None

    def _run(self):
        frame_interval = 1.0 / self.TARGET_FPS
        consecutive_failures = 0
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 60]

        while True:
            with self.lock:
                if not self.running:
                    break

            if not self.cap or not self.cap.isOpened():
                logger.info(f"Opening camera: {self.camera_url}")
                self.cap = self._open_capture()
                if not self.cap:
                    time.sleep(2)
                    with self.lock:
                        if self.viewers == 0 and time.monotonic() - self.last_viewer_at > self.IDLE_TIMEOUT_SECONDS:
                            self.running = False
                    continue
                consecutive_failures = 0

            loop_start = time.monotonic()
            ret, frame = self.cap.read()

            if not ret or frame is None:
                consecutive_failures += 1
                if consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
                    logger.error(f"Too many failures, releasing camera: {self.camera_url}")
                    try:
                        self.cap.release()
                    except Exception:
                        pass
                    self.cap = None
                time.sleep(0.1)
                continue

            consecutive_failures = 0
            self._frame_counter += 1

            raw = cv2.resize(frame, (480, 360))
            ok, buf = cv2.imencode('.jpg', raw, encode_param)
            raw_jpeg = buf.tobytes() if ok else None

            ai_jpeg = None
            if self._frame_counter % self.AI_EVERY_N_FRAMES == 0:
                annotated = doAI(frame)
                if annotated is not None:
                    ok, buf = cv2.imencode('.jpg', annotated, encode_param)
                    ai_jpeg = buf.tobytes() if ok else None

            with self.condition:
                if raw_jpeg is not None:
                    self.latest_jpeg = raw_jpeg
                    self.frame_version += 1
                if ai_jpeg is not None:
                    self.latest_ai_jpeg = ai_jpeg
                    self.ai_frame_version += 1
                self.condition.notify_all()

            with self.lock:
                if self.viewers == 0 and time.monotonic() - self.last_viewer_at > self.IDLE_TIMEOUT_SECONDS:
                    logger.info(f"No viewers for {self.IDLE_TIMEOUT_SECONDS}s, releasing camera: {self.camera_url}")
                    self.running = False

            elapsed = time.monotonic() - loop_start
            sleep_for = frame_interval - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        with self.condition:
            self.condition.notify_all()
        with camera_registry_lock:
            if camera_registry.get(self.camera_url) is self:
                camera_registry.pop(self.camera_url, None)
        logger.info(f"Camera stream stopped: {self.camera_url}")


camera_registry = {}
camera_registry_lock = threading.Lock()


def get_stream(camera_url):
    if not is_safe_camera_url(camera_url):
        logger.warning("Rejected unsafe camera URL: %s", camera_url)
        return None
    with camera_registry_lock:
        stream = camera_registry.get(camera_url)
        if stream is not None and stream.is_alive():
            return stream
        if stream is not None:
            camera_registry.pop(camera_url, None)
        if len(camera_registry) >= MAX_ACTIVE_CAMERAS:
            logger.warning(f"Max active cameras limit ({MAX_ACTIVE_CAMERAS}) reached")
            return None
        stream = CameraStream(camera_url)
        camera_registry[camera_url] = stream
        stream.start()
        return stream


def stop_stream(camera_url):
    with camera_registry_lock:
        stream = camera_registry.pop(camera_url, None)
    if stream:
        stream.stop()


def mjpeg_generator(camera_url, apply_ai):
    stream = get_stream(camera_url)
    if not stream:
        return
    stream.add_viewer()
    last_version = 0
    try:
        while True:
            frame, last_version = stream.get_frame(ai=apply_ai, last_version=last_version, timeout=15)
            if frame is None:
                logger.info(f"Stream ended for {camera_url}")
                break
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    finally:
        stream.remove_viewer()


@app.route('/logout', methods=['POST'])
@login_required
def logout():
    if not validate_csrf_token():
        return redirect(url_for('index'))
    # Camera streams are shared, global resources keyed by URL, not by session -
    # one user logging out must not kill a stream other logged-in users are
    # currently watching. Idle streams clean themselves up via IDLE_TIMEOUT_SECONDS.
    logout_user()
    session.clear()
    return redirect(url_for('index'))


@app.route('/api/user/add', methods=['POST'])
@admin_required
@limiter.limit("20 per minute")
def api_add_user():
    data = request.get_json(silent=True) or {}
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()
    role = data.get('role', 'user').strip()
    if not username or not password:
        return jsonify({'success': False, 'error': 'Username and password are required'}), 400
    if not is_safe_username(username):
        return jsonify({'success': False, 'error': 'Username must be 3-32 characters (letters, numbers, . _ -)'}), 400
    if not (8 <= len(password) <= 128):
        return jsonify({'success': False, 'error': 'Password must be between 8 and 128 characters'}), 400
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


@app.route('/api/user/remove', methods=['POST'])
@admin_required
@limiter.limit("20 per minute")
def api_remove_user():
    data = request.get_json(silent=True) or {}
    try:
        user_id = int(data.get('user_id'))
    except (TypeError, ValueError):
        return jsonify({'success': False, 'error': 'Invalid user ID'}), 400
    if user_id == current_user.id:
        return jsonify({'success': False, 'error': 'Cannot delete your own account'}), 400
    user = User.query.get(user_id)
    if user:
        # Grab their camera URLs before the cascade delete removes the rows, so
        # any streams still running for them can be shut down too.
        camera_urls = [cam.url for cam in user.cameras]
        username = user.username
        db.session.delete(user)
        db.session.commit()
        for url in camera_urls:
            stop_stream(url)
        logger.info(f"User removed: {username} (and {len(camera_urls)} camera(s))")
        return jsonify({'success': True, 'message': 'User removed successfully'})
    return jsonify({'success': False, 'error': 'User not found'}), 404


if __name__ == '__main__':
    host = os.environ.get('FLASK_RUN_HOST', '0.0.0.0')
    port = int(os.environ.get('FLASK_RUN_PORT', 8000))

    if os.environ.get('FLASK_ENV', 'production') == 'development':
        logger.info(f"Starting Flask dev server (threaded) on {host}:{port}")
        app.run(debug=False, host=host, port=port, threaded=True)
    else:
        try:
            from waitress import serve
            logger.info(f"Starting production server (waitress) on {host}:{port}")
            serve(app, host=host, port=port, threads=8)
        except ImportError:
            logger.warning(
                "waitress not installed - falling back to Flask's built-in server, "
                "which is not recommended for production. Run `pip install waitress`."
            )
            app.run(debug=False, host=host, port=port, threaded=True)
