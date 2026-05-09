from flask import Flask, render_template, request, Response, session, redirect, url_for, jsonify
import cv2
from ultralytics import YOLO
import threading
import time
import logging
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("DEBUG: Starting Flask app initialization")

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this to a secure secret key

print("DEBUG: Loading YOLO model")
try:
    model = YOLO('yolo11n.pt')  # Load once
    print("DEBUG: YOLO model loaded successfully")
except Exception as e:
    print(f"DEBUG: Error loading YOLO model: {e}")
    model = None

print("DEBUG: Flask app created")

# Global variables for camera management
camera_lock = threading.Lock()
active_cameras = {}

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

# Dashboard with camera input
@app.route('/dashboard')
def dashboard():
    if not session.get('logged_in'):
        return redirect(url_for('index'))
    
    # Initialize with default camera if no cameras exist or session is corrupted
    if 'cameras' not in session or not isinstance(session['cameras'], dict):
        session['cameras'] = {'Default Camera': 'webcam:0'}
        session.modified = True
    elif not session['cameras']:
        session['cameras'] = {'Default Camera': 'webcam:0'}
        session.modified = True
    
    return render_template('index.html')

# Error handler for 500 errors
@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Server Error: {error}")
    return "Internal Server Error. Please check logs.", 500

@app.errorhandler(404)
def not_found_error(error):
    return "Page Not Found", 404

# Save camera configuration
@app.route('/save_camera', methods=['POST'])
def save_camera():
    if not session.get('logged_in'):
        return redirect(url_for('index'))
    
    camera_name = request.form.get('camera_name')
    rtsp_url = request.form.get('rtsp_url')
    
    # Store in session (you might want to use a database in production)
    if 'cameras' not in session:
        session['cameras'] = {}
    
    session['cameras'][camera_name] = rtsp_url
    session.modified = True
    
    return redirect(url_for('dashboard'))

# Add default camera
@app.route('/add_default_camera', methods=['POST'])
def add_default_camera():
    if not session.get('logged_in'):
        return redirect(url_for('index'))
    
    camera_name = request.form.get('camera_name', 'Default Camera')
    camera_index = request.form.get('camera_index', '0')
    
    if 'cameras' not in session:
        session['cameras'] = {}
    
    # Store as a special format to distinguish from RTSP
    session['cameras'][camera_name] = f"webcam:{camera_index}"
    session.modified = True
    
    return redirect(url_for('dashboard'))

# Remove camera
@app.route('/remove_camera', methods=['POST'])
def remove_camera():
    if not session.get('logged_in'):
        return redirect(url_for('index'))
    
    camera_name = request.form.get('camera_name')
    
    if 'cameras' in session and camera_name in session['cameras']:
        del session['cameras'][camera_name]
        
        # If the removed camera was selected, clear selection
        if session.get('selected_camera') == camera_name:
            session['selected_camera'] = None
        
        session.modified = True
    
    return redirect(url_for('dashboard'))

# Get camera status
@app.route('/camera_status')
def camera_status():
    if not session.get('logged_in'):
        return jsonify({'status': 'error', 'message': 'Not authenticated'})
    
    camera_url = session.get('cameras', {}).get(session.get('selected_camera', ''), '')
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
    
    cameras = session.get('cameras')
    if not isinstance(cameras, dict):
        logger.error(f"Invalid cameras in session: {type(cameras)}")
        session['cameras'] = {'Default Camera': 'webcam:0'}
        cameras = session['cameras']
    
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

print("DEBUG: Test route registered")

# 🎥 Stream Routes
@app.route('/golive')
def video_feed():
    if not session.get('logged_in'):
        return redirect(url_for('index'))
    
    camera_url = session.get('cameras', {}).get(session.get('selected_camera', ''), '')
    if not camera_url:
        return "No camera selected or configured", 400
        
    return Response(stream_frames(camera_url, apply_ai=False),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/goai')
def video_feedai():
    if not session.get('logged_in'):
        return redirect(url_for('index'))
    
    camera_url = session.get('cameras', {}).get(session.get('selected_camera', ''), '')
    if not camera_url:
        return "No camera selected or configured", 400
        
    return Response(stream_frames(camera_url, apply_ai=True),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# New parameterized routes for multi-camera support
@app.route('/video_feed/<camera_name>')
def video_feed_named(camera_name):
    if not session.get('logged_in'):
        return redirect(url_for('index'))
    
    cameras = session.get('cameras', {})
    camera_url = cameras.get(camera_name)
    
    if not camera_url:
        return f"Camera '{camera_name}' not found", 404
        
    return Response(stream_frames(camera_url, apply_ai=False),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_ai/<camera_name>')
def video_feed_ai_named(camera_name):
    if not session.get('logged_in'):
        return redirect(url_for('index'))
    
    cameras = session.get('cameras', {})
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
        
        # Set some properties to help with streams
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Try to open the stream with a timeout
        start_time = time.time()
        while not cap.isOpened() and time.time() - start_time < 10:  # 10 second timeout
            time.sleep(0.1)
            
        if cap.isOpened():
            active_cameras[camera_url] = cap
            return cap
        else:
            logger.error(f"Failed to open camera stream: {camera_url}")
            return None

# 🔁 Improved Streaming Logic with Error Handling
def stream_frames(camera_url, apply_ai=False):
    reconnect_attempts = 0
    max_reconnect_attempts = 5
    
    while True:
        try:
            cap = get_camera(camera_url)
            if cap is None:
                # Generate a placeholder image when camera is not available
                placeholder = generate_placeholder_image("Camera not available")
                _, buffer = cv2.imencode('.jpg', placeholder)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                time.sleep(2)  # Wait before trying again
                continue
                
            ret, frame = cap.read()
            
            if not ret:
                reconnect_attempts += 1
                logger.warning(f"Failed to read frame, attempt {reconnect_attempts}/{max_reconnect_attempts}")
                
                if reconnect_attempts >= max_reconnect_attempts:
                    # Generate a placeholder image
                    placeholder = generate_placeholder_image("Reconnecting to camera...")
                    _, buffer = cv2.imencode('.jpg', placeholder)
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    
                    # Reset the camera connection
                    with camera_lock:
                        if camera_url in active_cameras:
                            try:
                                active_cameras[camera_url].release()
                            except:
                                pass
                            active_cameras.pop(camera_url, None)
                    
                    reconnect_attempts = 0
                    time.sleep(2)  # Wait before reconnecting
                    continue
                else:
                    # Continue with the same connection
                    time.sleep(0.1)
                    continue
            else:
                # Reset reconnect attempts on successful frame read
                reconnect_attempts = 0
                
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
            logger.error(f"Error in stream_frames: {e}")
            # Generate a placeholder image on error
            placeholder = generate_placeholder_image(f"Error: {str(e)}")
            _, buffer = cv2.imencode('.jpg', placeholder)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(2)  # Wait before trying again

# Generate placeholder image when camera is not available
def generate_placeholder_image(message):
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(img, message, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, "Please check camera connection", (50, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img

# Logout
@app.route('/logout')
def logout():
    # Release all camera resources on logout
    with camera_lock:
        for camera_url, cap in active_cameras.items():
            try:
                cap.release()
            except:
                pass
        active_cameras.clear()
    
    session.clear()
    return redirect(url_for('index'))

# 🚀 Launch
if __name__ == '__main__':
    print("DEBUG: About to start Flask app")
    print(f"DEBUG: Registered routes: {[rule.rule for rule in app.url_map.iter_rules()]}")
    app.run(host='0.0.0.0', port=8000, threaded=True, debug=True)