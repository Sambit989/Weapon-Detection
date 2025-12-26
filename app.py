from flask import Flask, request, jsonify, render_template_string, Response
from flask_cors import CORS
import cv2
import numpy as np
from ultralytics import YOLO
import threading
import time
import os
from datetime import datetime

import requests
import json
from collections import deque
import queue

app = Flask(__name__)
CORS(app)

class CriminalDetectionSystem:
    def __init__(self):
        # Load YOUR trained model
        try:
            self.model = YOLO(r'best2.pt')
            print("✅ Your trained model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading your model: {e}")
            self.model = None
        
        # SMS Configuration

        self.police_numbers = ['+917848084152', '+919117597217']
        
        # Detection settings - OPTIMIZED
        self.confidence_threshold = 0.6
        self.img_size = 416  # Reduced from 640 for faster processing
        
        # System state
        self.detection_active = False
        self.total_detections = 0
        self.alerts_sent = 0
        self.last_alert_time = 0
        self.alert_cooldown = 5
        self.current_location = None
        
        # PERFORMANCE OPTIMIZATION SETTINGS
        self.process_every_n_frames = 5  # Process every 5th frame (adjustable)
        self.frame_skip_for_display = 2  # Skip frames for display stream
        self.max_queue_size = 3  # Keep queue small to reduce latency
        self.frame_resize_factor = 0.7  # Resize frames for faster processing
        
        # Remote video feed URL
        self.remote_video_url = "http://10.81.187.79:5000/video_feed"
        
        # FPS tracking
        self.fps_counter = deque(maxlen=30)
        self.last_fps_time = time.time()
        self.current_fps = 0
        
        # Create directories
        self.footage_dir = "detected_footage"
        self.create_directories()
        self.get_current_location()
        
    def create_directories(self):
        if not os.path.exists(self.footage_dir):
            os.makedirs(self.footage_dir)
            print(f"📁 Created footage directory: {self.footage_dir}")
    
    def get_current_location(self):
        try:
            response = requests.get('http://ip-api.com/json/', timeout=5)
            data = response.json()
            
            if data['status'] == 'success':
                self.current_location = {
                    'latitude': data['lat'],
                    'longitude': data['lon'],
                    'city': data['city'],
                    'region': data['regionName'],
                    'country': data['country']
                }
                print(f"📍 Location: {data['city']}, {data['regionName']}")
            else:
                self.current_location = {
                    'latitude': 22.2587,
                    'longitude': 84.8854,
                    'city': 'Rourkela',
                    'region': 'Odisha',
                    'country': 'India'
                }
        except Exception as e:
            print(f"❌ Location error: {e}")
            self.current_location = {
                'latitude': 22.2587,
                'longitude': 84.8854,
                'city': 'Rourkela',
                'region': 'Odisha',
                'country': 'India'
            }
    
    def save_detection_footage(self, frame, weapon_type, confidence):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.footage_dir}/detection_{weapon_type}{timestamp}{confidence:.2f}.jpg"
            
            overlay_frame = frame.copy()
            text = f"WEAPON: {weapon_type} ({confidence:.2f})"
            cv2.putText(overlay_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            success = cv2.imwrite(filename, overlay_frame)
            if success:
                print(f"💾 Saved: {filename}")
                return filename
            return None
        except Exception as e:
            print(f"❌ Save error: {e}")
            return None
    
    def detect_objects(self, frame):
        if self.model is None:
            return []
            
        try:
            # Resize frame for faster processing
            h, w = frame.shape[:2]
            new_w = int(w * self.frame_resize_factor)
            new_h = int(h * self.frame_resize_factor)
            resized = cv2.resize(frame, (new_w, new_h))
            
            # Run detection on resized frame
            results = self.model.predict(
                source=resized, 
                imgsz=self.img_size, 
                conf=self.confidence_threshold, 
                verbose=False,
                device='cpu',  # Specify CPU explicitly
                half=False  # Disable half precision for stability
            )
            
            detections = []
            scale_x = w / new_w
            scale_y = h / new_h
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        class_name = self.model.names[cls_id]
                        
                        # Scale bounding boxes back to original size
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                        y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                        
                        detection = {
                            'class_name': class_name,
                            'confidence': conf,
                            'bbox': [x1, y1, x2, y2]
                        }
                        detections.append(detection)
                        
            return detections
            
        except Exception as e:
            print(f"Detection error: {e}")
            return []
    

    
    def should_alert(self):
        return (time.time() - self.last_alert_time) > self.alert_cooldown
    
    def update_fps(self):
        current_time = time.time()
        self.fps_counter.append(current_time)
        
        if len(self.fps_counter) > 1:
            time_diff = self.fps_counter[-1] - self.fps_counter[0]
            if time_diff > 0:
                self.current_fps = len(self.fps_counter) / time_diff

# Initialize system
detector = CriminalDetectionSystem()

# Global variables - OPTIMIZED WITH QUEUE
frame_queue = queue.Queue(maxsize=detector.max_queue_size)
detection_thread = None
current_detections = []
frame_lock = threading.Lock()
latest_frame = None
stop_threads = False

def fetch_remote_stream_optimized():
    """Optimized stream fetcher with frame dropping"""
    global frame_queue, stop_threads
    
    print(f"🌐 Connecting to: {detector.remote_video_url}")
    
    try:
        stream = requests.get(detector.remote_video_url, stream=True, timeout=10)
        
        if stream.status_code != 200:
            print(f"❌ Connection failed: {stream.status_code}")
            return
        
        print("✅ Connected to remote feed!")
        
        bytes_data = bytes()
        frame_count = 0
        
        for chunk in stream.iter_content(chunk_size=4096):  # Larger chunks
            if stop_threads or not detector.detection_active:
                break
                
            bytes_data += chunk
            
            a = bytes_data.find(b'\xff\xd8')
            b = bytes_data.find(b'\xff\xd9')
            
            if a != -1 and b != -1:
                jpg = bytes_data[a:b+2]
                bytes_data = bytes_data[b+2:]
                
                # Decode frame
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                
                if frame is not None:
                    frame_count += 1
                    
                    # Drop frames if queue is full (reduces latency)
                    if frame_queue.full():
                        try:
                            frame_queue.get_nowait()  # Remove old frame
                        except queue.Empty:
                            pass
                    
                    try:
                        frame_queue.put(frame, block=False)
                    except queue.Full:
                        pass  # Skip this frame
                    
    except Exception as e:
        print(f"❌ Stream error: {e}")
    finally:
        print("📡 Stream fetcher stopped")

def detection_loop_optimized():
    """Optimized detection loop"""
    global current_detections, latest_frame, stop_threads
    
    print("🎥 Starting optimized detection...")
    
    frame_count = 0
    last_process_time = time.time()
    
    while detector.detection_active and not stop_threads:
        try:
            # Get frame from queue with timeout
            try:
                frame = frame_queue.get(timeout=1)
            except queue.Empty:
                continue
            
            # Update display frame
            with frame_lock:
                latest_frame = frame.copy()
            
            # Update FPS
            detector.update_fps()
            
            frame_count += 1
            
            # Process only every Nth frame for detection
            if frame_count % detector.process_every_n_frames == 0:
                process_start = time.time()
                
                detections = detector.detect_objects(frame)
                current_detections = detections
                
                process_time = time.time() - process_start
                print(f"⚡ Detection time: {process_time*1000:.1f}ms, FPS: {detector.current_fps:.1f}")
                
                if detections:
                    detector.total_detections += 1
                    
                    if detector.should_alert():
                        for det in detections:
                            weapon_name = det['class_name']
                            confidence = det['confidence'] * 100
                            
                            print(f"🚨 {weapon_name} ({confidence:.1f}%)")
                            
                            # Save footage in background thread to avoid blocking
                            threading.Thread(
                                target=detector.save_detection_footage,
                                args=(frame, weapon_name, confidence),
                                daemon=True
                            ).start()
                            
                            # Send alert in background

                            
                            break
            
        except Exception as e:
            print(f"❌ Detection loop error: {e}")
            time.sleep(0.1)
    
    print("🎥 Detection loop stopped")

def generate_frames_optimized():
    """Optimized frame generator for display"""
    global latest_frame, current_detections
    
    frame_count = 0
    
    while detector.detection_active:
        if latest_frame is None:
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Connecting...", (200, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', placeholder, [cv2.IMWRITE_JPEG_QUALITY, 70])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.1)
            continue
        
        frame_count += 1
        
        # Skip frames for display to reduce bandwidth
        if frame_count % detector.frame_skip_for_display != 0:
            time.sleep(0.01)
            continue
        
        with frame_lock:
            frame = latest_frame.copy()
        
        # Draw detections
        for det in current_detections:
            bbox = det['bbox']
            name = det['class_name']
            conf = det['confidence']
            
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
            
            label = f"{name}: {conf:.2f}"
            cv2.rectangle(frame, (bbox[0], bbox[1]-25), (bbox[0]+150, bbox[1]), (0, 0, 255), -1)
            cv2.putText(frame, label, (bbox[0]+5, bbox[1]-8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add compact overlay
        info_bg = frame.copy()
        cv2.rectangle(info_bg, (0, 0), (250, 100), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, info_bg, 0.3, 0)
        
        cv2.putText(frame, f"FPS: {detector.current_fps:.1f}", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Detections: {detector.total_detections}", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, f"Alerts: {detector.alerts_sent}", (10, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Encode with lower quality for faster streaming
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        time.sleep(0.02)  # ~50 FPS max for display

@app.route('/')
def index():
    return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <title>Optimized Detection System</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #1a1a1a; color: white; }
        .container { max-width: 1400px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 30px; }
        .header h1 { color: #ff4444; font-size: 2.5rem; margin: 0; }
        .optimization-badge { background: #4CAF50; color: white; padding: 5px 15px; border-radius: 20px; font-size: 0.9rem; margin: 5px; }
        .main { display: grid; grid-template-columns: 2fr 1fr; gap: 30px; }
        .video-section { background: #333; padding: 20px; border-radius: 10px; }
        .control-section { background: #333; padding: 20px; border-radius: 10px; }
        .video-container { width: 100%; height: 480px; background: black; border-radius: 10px; overflow: hidden; position: relative; }
        .video-stream { width: 100%; height: 100%; object-fit: contain; }
        .controls { margin-top: 20px; text-align: center; }
        .btn { padding: 15px 30px; margin: 10px; border: none; border-radius: 5px; font-size: 16px; cursor: pointer; font-weight: bold; }
        .btn-start { background: #4CAF50; color: white; }
        .btn-stop { background: #f44336; color: white; }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .status { padding: 20px; margin: 20px 0; border-radius: 10px; }
        .status-active { background: #4CAF50; }
        .status-inactive { background: #666; }
        .stats { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; margin-bottom: 20px; }
        .stat { background: #444; padding: 15px; border-radius: 10px; text-align: center; }
        .stat-number { font-size: 2rem; font-weight: bold; color: #4CAF50; }
        .stat-label { font-size: 0.9rem; color: #aaa; }
        .optimization-info { background: #2a4a2a; padding: 15px; border-radius: 10px; margin-bottom: 20px; border-left: 4px solid #4CAF50; }
        .optimization-info h4 { margin: 0 0 10px 0; color: #4CAF50; }
        .optimization-info ul { margin: 5px 0; padding-left: 20px; }
        .optimization-info li { margin: 5px 0; font-size: 0.9rem; }
        .loading { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); color: #666; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>⚡ OPTIMIZED DETECTION SYSTEM</h1>
            <div>
                <span class="optimization-badge">🚀 Low Latency</span>
                <span class="optimization-badge">⚡ High FPS</span>
                <span class="optimization-badge">🎯 Smart Frame Skipping</span>
            </div>
            <p>AI-Powered Weapon Detection with Performance Optimizations</p>
        </div>
        
        <div class="main">
            <div class="video-section">
                <h2>📹 Live Feed (Optimized)</h2>
                <div class="video-container">
                    <img id="videoStream" class="video-stream" src="/video_feed" style="display: none;">
                    <div id="noVideo" class="loading">Click Start Detection</div>
                </div>
                <div class="controls">
                    <button id="startBtn" class="btn btn-start" onclick="startDetection()">🎯 START</button>
                    <button id="stopBtn" class="btn btn-stop" onclick="stopDetection()" disabled>⏹ STOP</button>
                </div>
            </div>
            
            <div class="control-section">
                <div id="status" class="status status-inactive">
                    <h3>Status: <span id="statusText">OFFLINE</span></h3>
                    <p id="statusDetails">Ready to start</p>
                </div>
                
                <div class="stats">
                    <div class="stat">
                        <div id="fpsCount" class="stat-number">0</div>
                        <div class="stat-label">FPS</div>
                    </div>
                    <div class="stat">
                        <div id="detectionCount" class="stat-number">0</div>
                        <div class="stat-label">Detections</div>
                    </div>
                    <div class="stat">
                        <div id="alertCount" class="stat-number">0</div>
                        <div class="stat-label">Alerts</div>
                    </div>
                </div>
                
                <div class="optimization-info">
                    <h4>⚡ Performance Optimizations Active</h4>
                    <ul>
                        <li>Frame skipping: Process every 5th frame</li>
                        <li>Model input: 416px (faster inference)</li>
                        <li>Frame resize: 70% for detection</li>
                        <li>Display throttling: Reduced bandwidth</li>
                        <li>Queue size: 3 frames (low latency)</li>
                        <li>Background SMS: Non-blocking alerts</li>
                    </ul>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let statsInterval;
        
        function startDetection() {
            fetch('/start', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        document.getElementById('startBtn').disabled = true;
                        document.getElementById('stopBtn').disabled = false;
                        document.getElementById('videoStream').style.display = 'block';
                        document.getElementById('noVideo').style.display = 'none';
                        document.getElementById('status').className = 'status status-active';
                        document.getElementById('statusText').textContent = 'ACTIVE';
                        document.getElementById('statusDetails').textContent = 'Optimized detection running';
                        startStatsUpdate();
                        
                        setTimeout(() => {
                            document.getElementById('videoStream').src = '/video_feed?' + new Date().getTime();
                        }, 500);
                    }
                });
        }
        
        function stopDetection() {
            fetch('/stop', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        document.getElementById('startBtn').disabled = false;
                        document.getElementById('stopBtn').disabled = true;
                        document.getElementById('videoStream').style.display = 'none';
                        document.getElementById('noVideo').style.display = 'block';
                        document.getElementById('status').className = 'status status-inactive';
                        document.getElementById('statusText').textContent = 'OFFLINE';
                        document.getElementById('statusDetails').textContent = 'Detection stopped';
                        stopStatsUpdate();
                    }
                });
        }
        
        function startStatsUpdate() {
            statsInterval = setInterval(updateStats, 500);
        }
        
        function stopStatsUpdate() {
            if (statsInterval) clearInterval(statsInterval);
        }
        
        function updateStats() {
            fetch('/stats')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('fpsCount').textContent = data.fps.toFixed(1);
                    document.getElementById('detectionCount').textContent = data.detections;
                    document.getElementById('alertCount').textContent = data.alerts;
                });
        }
    </script>
</body>
</html>
    ''')

@app.route('/start', methods=['POST'])
def start_detection():
    global detection_thread, stop_threads
    
    if detector.detection_active:
        return jsonify({'success': False, 'message': 'Already running'})
    
    try:
        stop_threads = False
        detector.detection_active = True
        
        # Start stream fetcher thread
        stream_thread = threading.Thread(target=fetch_remote_stream_optimized, daemon=True)
        stream_thread.start()
        
        # Start detection thread
        detection_thread = threading.Thread(target=detection_loop_optimized, daemon=True)
        detection_thread.start()
        
        print("🚨 Optimized detection started!")
        return jsonify({'success': True, 'message': 'Detection started'})
        
    except Exception as e:
        print(f"❌ Start error: {e}")
        detector.detection_active = False
        stop_threads = True
        return jsonify({'success': False, 'message': str(e)})

@app.route('/stop', methods=['POST'])
def stop_detection():
    global stop_threads
    
    detector.detection_active = False
    stop_threads = True
    
    # Clear queue
    while not frame_queue.empty():
        try:
            frame_queue.get_nowait()
        except queue.Empty:
            break
    
    print("🛑 Detection stopped!")
    return jsonify({'success': True, 'message': 'Detection stopped'})

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames_optimized(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def get_stats():
    return jsonify({
        'fps': detector.current_fps,
        'detections': detector.total_detections,
        'alerts': detector.alerts_sent
    })

if __name__ == '__main__':
    print("⚡ OPTIMIZED DETECTION SYSTEM")
    print("=" * 60)
    print("📡 Remote: http://10.151.43.79:5000/video_feed")
    print("⚡ Optimizations:")
    print("   - Process every 5th frame (5x faster)")
    print("   - 416px model input (1.5x faster inference)")
    print("   - 70% frame resize (2x faster)")
    print("   - Queue size: 3 (low latency)")
    print("   - Background SMS (non-blocking)")
    print("🌐 Interface: http://localhost:5001")
    print("=" * 60)
    
    # Run on different port to avoid conflict
    app.run(debug=False, host='0.0.0.0', port=5001, threaded=True)