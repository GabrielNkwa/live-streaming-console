from flask import Flask, render_template, request, Response, session, redirect, url_for, jsonify, abort, send_from_directory
from werkzeug.middleware.proxy_fix import ProxyFix
import cv2
from ultralytics import YOLO
import threading
import time
import logging
import numpy as np
import os
import json
import secrets
import re
from datetime import timedelta
from urllib.parse import urlparse

# Set up logging
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
)

if os.environ.get('USE_PROXY_FIX') == '1':
    app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

ASSETS_FOLDER = os.path.join(app.root_path, 'static', 'assets')
os.makedirs(ASSETS_FOLDER, exist_ok=True)

logger.info("Loading YOLO model")
try:
    model = YOLO('yolo11n.pt')  # Load once
    logger.info("YOLO model loaded successfully")
except Exception as e:
    logger.error(f"Error loading YOLO model: {e}")
    model = None

logger.info("Flask app created")

# Global variables for camera management
camera_lock = threading.Lock()
active_cameras = {}
MAX_ACTIVE_CAMERAS = 10  # Limit total active camera connections

# Camera file path
CAMERAS_FILE = os.path.join('static', 'cams.txt')

def ensure_cameras_file_exists():
    """Ensure the cameras file exists"""
    os.makedirs('static', exist_ok=True)
    if not os.path.exists(CAMERAS_FILE):
        # Create default camera
        with open(CAMERAS_FILE, 'w') as f:
            f.write('Default Camera,webcam:0\n')
        logger.info(f"Created default cameras file: {CAMERAS_FILE}")


def generate_csrf_token():
    token = session.get('csrf_token')
    if not token:
        token = secrets.token_urlsafe(32)
        session['csrf_token'] = token
    return token


def is_safe_camera_name(camera_name):
    return bool(re.fullmatch(r"[A-Za-z0-9 _\-]{1,64}", camera_name))


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

    hostname = parsed.hostname.lower()
    if hostname in ('127.0.0.1', 'localhost', '0.0.0.0'):
        return False
    if hostname.startswith('169.254.'):
        return False
    return True


def is_admin():
    return session.get('logged_in') and session.get('role') == 'admin'


def is_authenticated():
    return session.get('logged_in')


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


def load_cameras():
    """Load cameras from cams.txt file"""
    ensure_cameras_file_exists()
    cameras = {}
    try:
        with open(CAMERAS_FILE, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split(',', 1)
                    if len(parts) == 2:
                        name, url = parts
                        cameras[name.strip()] = url.strip()
        logger.info(f"Loaded {len(cameras)} cameras from file")
    except Exception as e:
        logger.error(f"Error loading cameras: {e}")
    return cameras

def save_cameras(cameras):
    """Save cameras to cams.txt file"""
    try:
        ensure_cameras_file_exists()
        with open(CAMERAS_FILE, 'w') as f:
            for name, url in cameras.items():
                f.write(f'{name},{url}\n')
        logger.info(f"Saved {len(cameras)} cameras to file")
        return True
    except Exception as e:
        logger.error(f"Error saving cameras: {e}")
        return False

def add_camera(name, url):
    """Add a new camera to the file"""
    cameras = load_cameras()
    cameras[name] = url
    return save_cameras(cameras)

def remove_camera_file(name):
    """Remove a camera from the file"""
    cameras = load_cameras()
    if name in cameras:
        del cameras[name]
        return save_cameras(cameras)
    return False

# Initialize cameras on startup
ensure_cameras_file_exists()

# 🔐 Login Page
@app.route('/')
def index():
    return render_template('login.html')

ADMIN_USERNAME = os.environ.get('APP_ADMIN_USER', 'admin')
ADMIN_PASSWORD = os.environ.get('APP_ADMIN_PASS', 'admin')
if ADMIN_USERNAME == 'admin' and ADMIN_PASSWORD == 'admin':
    logger.warning('Using default admin credentials. Set APP_ADMIN_USER and APP_ADMIN_PASS in production.')

@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username', '').strip()
    password = request.form.get('password', '').strip()
    if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        session.clear()
        session['logged_in'] = True
        session['role'] = 'admin'
        session.permanent = True
        generate_csrf_token()
        return redirect(url_for('dashboard'))
    return render_template('login.html', error='Invalid credentials')

# Dashboard with live streaming
@app.route('/dashboard')
def dashboard():
    if not is_authenticated():
        return redirect(url_for('index'))
    
    cameras = load_cameras()
    selected_camera = session.get('selected_camera')
    if selected_camera not in cameras:
        session.pop('selected_camera', None)
        selected_camera = None
    return render_template('index.html', cameras=cameras, selected_camera=selected_camera)

# Admin Panel for camera management
@app.route('/admin')
def admin():
    if not is_admin():
        return redirect(url_for('index'))
    
    cameras = load_cameras()
    return render_template('admin.html', cameras=cameras)

# Error handler for 500 errors
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
    # Add HSTS header (Strict-Transport-Security)
    if request.is_secure or os.environ.get('FLASK_ENV', 'production') == 'production':
        response.headers.setdefault('Strict-Transport-Security', 'max-age=31536000; includeSubDomains')
    return response

# Save camera configuration (API endpoint for admin panel)
@app.route('/api/camera/add', methods=['POST'])
def api_add_camera():
    if not is_admin():
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401
    
    data = request.get_json(silent=True) or {}
    camera_name = data.get('camera_name', '').strip()
    camera_url = data.get('camera_url', '').strip()
    
    if not camera_name or not camera_url:
        return jsonify({'success': False, 'error': 'Camera name and URL are required'}), 400
    if not is_safe_camera_name(camera_name):
        return jsonify({'success': False, 'error': 'Invalid camera name'}), 400
    if not is_safe_camera_url(camera_url):
        return jsonify({'success': False, 'error': 'Invalid or unsupported camera URL'}), 400
    
    cameras = load_cameras()
    if camera_name in cameras:
        return jsonify({'success': False, 'error': 'Camera name already exists'}), 400
    
    if add_camera(camera_name, camera_url):
        logger.info(f"Camera added: {camera_name}")
        return jsonify({'success': True, 'message': 'Camera added successfully'})
    else:
        return jsonify({'success': False, 'error': 'Failed to add camera'}), 500

# Remove camera
@app.route('/api/camera/remove', methods=['POST'])
def api_remove_camera():
    if not is_admin():
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401
    
    data = request.get_json(silent=True) or {}
    camera_name = data.get('camera_name', '').strip()
    
    if not camera_name or not is_safe_camera_name(camera_name):
        return jsonify({'success': False, 'error': 'Camera name is required and must be valid'}), 400
    
    # First, get the camera URL from the cameras file
    cameras = load_cameras()
    camera_url = cameras.get(camera_name)
    
    if remove_camera_file(camera_name):
        if session.get('selected_camera') == camera_name:
            session['selected_camera'] = None
            session.modified = True
        
        # Release camera from active_cameras if it's open
        if camera_url:
            with camera_lock:
                if camera_url in active_cameras:
                    try:
                        cap = active_cameras.pop(camera_url)
                        cap.release()
                        logger.info(f"Released active camera: {camera_url}")
                    except Exception as e:
                        logger.error(f"Error releasing camera {camera_url}: {e}")
        
        logger.info(f"Camera removed: {camera_name}")
        return jsonify({'success': True, 'message': 'Camera removed successfully'})
    else:
        return jsonify({'success': False, 'error': 'Camera not found'}), 404

# Get camera status
@app.route('/camera_status')
def camera_status():
    if not is_authenticated():
        return jsonify({'status': 'error', 'message': 'Not authenticated'})
    
    cameras = load_cameras()
    selected = session.get('selected_camera')
    camera_url = cameras.get(selected, '')
    
    if not camera_url or not is_safe_camera_url(camera_url):
        return jsonify({'status': 'error', 'message': 'No valid camera selected'})
    
    if camera_url.startswith('webcam:'):
        camera_index = int(camera_url.split(':', 1)[1])
        cap = cv2.VideoCapture(camera_index)
    else:
        cap = cv2.VideoCapture(camera_url)
    
    if cap.isOpened():
        ret, frame = cap.read()
        cap.release()
        if ret:
            return jsonify({'status': 'success', 'message': 'Camera connected'})
        else:
            return jsonify({'status': 'error', 'message': 'Camera connected but no frame received'})
    else:
        return jsonify({'status': 'error', 'message': 'Failed to connect to camera'})

# Select camera for streaming
@app.route('/select_camera', methods=['POST'])
def select_camera():
    if not is_authenticated():
        logger.warning("Unauthorized access attempt to select_camera")
        return redirect(url_for('index'))
    
    selected_camera = request.form.get('selected_camera', '').strip()
    logger.info(f"Camera selection request: {selected_camera}")
    
    cameras = load_cameras()
    
    if selected_camera and selected_camera in cameras:
        session['selected_camera'] = selected_camera
        session.modified = True
        logger.info(f"Camera selected successfully: {selected_camera}")
    else:
        logger.warning(f"Invalid camera selection: {selected_camera}")
    
    return redirect(url_for('dashboard'))

# Test route
@app.route('/test')
def test():
    return "Test route works!"

logger.info("Test route registered")

# 🎥 Stream Routes
@app.route('/golive')
def video_feed():
    if not is_authenticated():
        return redirect(url_for('index'))
    
    cameras = load_cameras()
    camera_url = cameras.get(session.get('selected_camera', ''), '')
    if not camera_url or not is_safe_camera_url(camera_url):
        return "No camera selected or configured", 400
        
    return Response(stream_frames(camera_url, apply_ai=False),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/goai')
def video_feedai():
    if not is_authenticated():
        return redirect(url_for('index'))
    
    cameras = load_cameras()
    camera_url = cameras.get(session.get('selected_camera', ''), '')
    if not camera_url or not is_safe_camera_url(camera_url):
        return "No camera selected or configured", 400
        
    return Response(stream_frames(camera_url, apply_ai=True),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# New parameterized routes for multi-camera support
@app.route('/video_feed/<camera_name>')
def video_feed_named(camera_name):
    if not is_authenticated():
        return redirect(url_for('index'))
    
    cameras = load_cameras()
    camera_url = cameras.get(camera_name)
    if not camera_url or not is_safe_camera_url(camera_url):
        return f"Camera '{camera_name}' not found", 404
    
    return Response(stream_frames(camera_url, apply_ai=False),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_ai/<camera_name>')
def video_feed_ai_named(camera_name):
    if not is_authenticated():
        return redirect(url_for('index'))
    
    cameras = load_cameras()
    camera_url = cameras.get(camera_name)
    if not camera_url or not is_safe_camera_url(camera_url):
        return f"Camera '{camera_name}' not found", 404
    
    return Response(stream_frames(camera_url, apply_ai=True),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# 🧠 AI Detection
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

# Get camera instance with reconnection logic
def get_camera(camera_url):
    if not is_safe_camera_url(camera_url):
        logger.warning("Rejected unsafe camera URL: %s", camera_url)
        return None

    with camera_lock:
        if camera_url in active_cameras:
            cap = active_cameras[camera_url]
            if cap.isOpened():
                return cap
            try:
                cap.release()
            except Exception:
                pass
            active_cameras.pop(camera_url, None)

        # Check max active cameras limit
        if len(active_cameras) >= MAX_ACTIVE_CAMERAS:
            logger.warning(f"Max active cameras limit ({MAX_ACTIVE_CAMERAS}) reached")
            return None

        logger.info(f"Creating new capture for {camera_url}")
        if camera_url.startswith('webcam:'):
            camera_index = int(camera_url.split(':', 1)[1])
            # Try DSHOW first (Windows), fall back to default
            cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(camera_index)
        else:
            cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
            # Optimize for RTSP streams - reduce latency
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
            cap.set(cv2.CAP_PROP_FPS, 30)

        if cap.isOpened():
            # Set reasonable resolution for performance
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            active_cameras[camera_url] = cap
            return cap

        try:
            cap.release()
        except Exception:
            pass
        logger.error(f"Failed to open camera stream: {camera_url}")
        return None

# 🔁 Streaming Logic with Error Handling
def stream_frames(camera_url, apply_ai=False):
    """Stream video frames from a camera URL with optional AI detection"""
    cap = None
    try:
        cap = get_camera(camera_url)
        if not cap:
            logger.error(f"Failed to get camera: {camera_url}")
            time.sleep(0.5)
            return
        
        while True:
            # Read multiple frames quickly to clear buffer
            for _ in range(2):
                cap.read()
            
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.001)
                continue

            if apply_ai:
                frame = doAI(frame)
            else:
                # Resize for speed
                frame = cv2.resize(frame, (480, 360))

            if frame is None:
                time.sleep(0.001)
                continue

            # Aggressive JPEG compression for maximum speed
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
            _, buffer = cv2.imencode('.jpg', frame, encode_param)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    except Exception as e:
        logger.error(f"Error in stream_frames for {camera_url}: {e}")
        if cap:
            try:
                with camera_lock:
                    if camera_url in active_cameras:
                        active_cameras.pop(camera_url, None)
                cap.release()
            except Exception:
                pass
            

# Logout
@app.route('/logout', methods=['POST'])
def logout():
    """Clear session and release all camera resources"""
    if not is_authenticated() or not validate_csrf_token():
        return redirect(url_for('index'))

    with camera_lock:
        for camera_url, cap in active_cameras.items():
            try:
                cap.release()
            except Exception:
                pass
        active_cameras.clear()
    
    session.clear()
    return redirect(url_for('index'))

# Run the app
if __name__ == '__main__':
    app.run(
        debug=False,
        host=os.environ.get('FLASK_RUN_HOST', '0.0.0.0'),
        port=int(os.environ.get('FLASK_RUN_PORT', 8000))
    )
