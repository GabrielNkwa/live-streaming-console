from flask import Flask, render_template, request, Response, session, redirect, url_for, jsonify
import cv2
from ultralytics import YOLO
import threading
import time
import logging
import numpy as np
import requests
import os
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("Starting Flask app initialization")

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this to a secure secret key

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

@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username')
    password = request.form.get('password')
    if username == 'admin' and password == 'admin':
        session['logged_in'] = True
        return redirect(url_for('dashboard'))
    return render_template('login.html', error='Invalid credentials')

# Dashboard with live streaming
@app.route('/dashboard')
def dashboard():
    if not session.get('logged_in'):
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
    if not session.get('logged_in'):
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

# Save camera configuration (API endpoint for admin panel)
@app.route('/api/camera/add', methods=['POST'])
def api_add_camera():
    if not session.get('logged_in'):
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401
    
    data = request.get_json()
    camera_name = data.get('camera_name', '').strip()
    camera_url = data.get('camera_url', '').strip()
    
    if not camera_name or not camera_url:
        return jsonify({'success': False, 'error': 'Camera name and URL are required'}), 400
    
    # Check if camera already exists
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
    if not session.get('logged_in'):
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401
    
    data = request.get_json()
    camera_name = data.get('camera_name', '').strip()
    
    if not camera_name:
        return jsonify({'success': False, 'error': 'Camera name is required'}), 400
    
    if remove_camera_file(camera_name):
        # If the removed camera was selected, clear selection
        if session.get('selected_camera') == camera_name:
            session['selected_camera'] = None
            session.modified = True
        logger.info(f"Camera removed: {camera_name}")
        return jsonify({'success': True, 'message': 'Camera removed successfully'})
    else:
        return jsonify({'success': False, 'error': 'Camera not found'}), 404

# Get camera status
@app.route('/camera_status')
def camera_status():
    if not session.get('logged_in'):
        return jsonify({'status': 'error', 'message': 'Not authenticated'})
    
    cameras = load_cameras()
    selected = session.get('selected_camera')
    camera_url = cameras.get(selected, '')
    
    if not camera_url:
        return jsonify({'status': 'error', 'message': 'No camera selected'})
    
    # Handle webcam URLs
    if camera_url.startswith('webcam:'):
        camera_index = int(camera_url.split(':')[1])
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
    if not session.get('logged_in'):
        logger.warning("Unauthorized access attempt to select_camera")
        return redirect(url_for('index'))
    
    selected_camera = request.form.get('selected_camera')
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
    if not session.get('logged_in'):
        return redirect(url_for('index'))
    
    cameras = load_cameras()
    camera_url = cameras.get(session.get('selected_camera', ''), '')
    if not camera_url:
        return "No camera selected or configured", 400
        
    return Response(stream_frames(camera_url, apply_ai=False),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/goai')
def video_feedai():
    if not session.get('logged_in'):
        return redirect(url_for('index'))
    
    cameras = load_cameras()
    camera_url = cameras.get(session.get('selected_camera', ''), '')
    if not camera_url:
        return "No camera selected or configured", 400
        
    return Response(stream_frames(camera_url, apply_ai=True),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# New parameterized routes for multi-camera support
@app.route('/video_feed/<camera_name>')
def video_feed_named(camera_name):
    if not session.get('logged_in'):
        return redirect(url_for('index'))
    
    cameras = load_cameras()
    camera_url = cameras.get(camera_name)
    if not camera_url:
        return f"Camera '{camera_name}' not found", 404
    
    return Response(stream_frames(camera_url, apply_ai=False),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_ai/<camera_name>')
def video_feed_ai_named(camera_name):
    if not session.get('logged_in'):
        return redirect(url_for('index'))
    
    cameras = load_cameras()
    camera_url = cameras.get(camera_name)
    if not camera_url:
        return f"Camera '{camera_name}' not found", 404
    
    return Response(stream_frames(camera_url, apply_ai=True),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# 🧠 AI Detection
def doAI(img):
    try:
        # Resize for YOLO processing (model expects certain input size)
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
        # If AI fails, return the original image resized
        return cv2.resize(img, (640, 480))

# Get camera instance with reconnection logic
def get_camera(camera_url):
    with camera_lock:
        if camera_url in active_cameras:
            cap = active_cameras[camera_url]
            # Check if the capture is still working
            if cap.isOpened():
                return cap
            else:
                # Clean up the old capture
                try:
                    cap.release()
                except:
                    pass
                # Remove from active cameras
                active_cameras.pop(camera_url, None)
        
        # Create a new capture
        logger.info(f"Creating new capture for {camera_url}")
        
        # Handle webcam URLs
        if camera_url.startswith('webcam:'):
            camera_index = int(camera_url.split(':')[1])
            cap = cv2.VideoCapture(camera_index)
        else:
            cap = cv2.VideoCapture(camera_url)
        print("GOOOO")
        # Set some properties to help with streams
        # cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # cap.set(cv2.CAP_PROP_FPS, 30)
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("GOOOO2") 
        # Try to open the stream with a timeout
        # start_time = time.time()
        # while not cap.isOpened() and time.time() - start_time < 10:  # 10 second timeout
        #     time.sleep(0.1)
            
        # if cap.isOpened():
        #     active_cameras[camera_url] = cap
        #     return cap
        # else:
        #     logger.error(f"Failed to open camera stream: {camera_url}")
        #     return None
        return cap

# 🔁 Streaming Logic with Error Handling
def stream_frames(camera_url, apply_ai=False):
    """Stream video frames from a camera URL with optional AI detection"""
    reconnect_attempts = 0
    max_reconnect_attempts = 5
    
    while True:
        try:
            cap = get_camera(camera_url)
            if not cap:
                logger.error(f"Failed to get camera: {camera_url}")
                # time.sleep(2)
                continue
            
            ret, frame = cap.read()
            print("Gooooo3")
            
            
            # Resize for better performance (only if not applying AI)
            if apply_ai:
                frame = doAI(frame)
            else:
                frame = cv2.resize(frame, (640, 480))
            
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    
        except Exception as e:
            logger.error(f"Error in stream_frames for {camera_url}: {e}")
            

# Logout
@app.route('/logout')
def logout():
    """Clear session and release all camera resources"""
    with camera_lock:
        for camera_url, cap in active_cameras.items():
            try:
                cap.release()
            except:
                pass
        active_cameras.clear()
    
    session.clear()
    return redirect(url_for('index'))

# Run the app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
