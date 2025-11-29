from flask import Flask, render_template, request, jsonify, send_file, Response, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import cv2
import threading
import time
import os
import atexit
import base64
from datetime import datetime
import uuid
import logging
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import optimized camera system
try:
    from camera_system import get_system_instance, cleanup_system
    SIMULATION_MODE = False
    print("Optimized camera system loaded successfully")
except ImportError as e:
    print(f"Camera system not available: {e}")
    print("Running in simulation mode")
    SIMULATION_MODE = True

# Initialize Flask app with enhanced SocketIO
app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = 'ocr-processing-system-2025'
socketio = SocketIO(
    app, 
    cors_allowed_origins="*", 
    async_mode='threading',
    ping_timeout=60,
    ping_interval=25
)

# Reflect Node-RED webhook configuration (keeps UI/API in sync with camera_system)
NODE_RED_URL = os.getenv('NODE_RED_URL', 'http://localhost:1880/webhook/screenshot')

# Configure enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Enhanced streaming control
streaming = False
stream_thread = None
stream_lock = threading.Lock()
connected_clients = 0

# Enhanced system state with performance tracking
system_state = {
    'running': False,
    'paused': False,
    'current_run_id': None,
    'service_provider': None,
    'image_count': 0,
    'total_runs': 0,
    'last_image_time': None,
    'error_message': None,
    'start_time': None,
    'last_image_path': None,
    'performance': {
        'avg_capture_time': 0,
        'avg_upload_time': 0,
        'images_per_minute': 0,
        'stream_fps': 0
    }
}

camera_system = None

def status_callback(status_data):
    """Enhanced callback for comprehensive logging and error forwarding"""
    global system_state
    
    # Handle different types of status updates
    callback_type = status_data.get('type', 'status_update')
    
    if callback_type == 'log_update':
        # Forward log messages to frontend
        socketio.emit('log_update', {
            'message': status_data.get('log_message', ''),
            'level': status_data.get('level', 'info'),
            'timestamp': status_data.get('timestamp', ''),
            'full_message': status_data.get('full_message', ''),
            'performance': status_data.get('performance', {}),
            'system_status': status_data.get('system_status', {})
        })
        
        # Update local system state
        if 'system_status' in status_data:
            system_state.update(status_data['system_status'])
            
        # Log to backend console with level formatting
        level = status_data.get('level', 'info').upper()
        message = status_data.get('log_message', '')
        timestamp = status_data.get('timestamp', '')
        print(f"[{timestamp}] [{level}] {message}")
        
    elif callback_type == 'error_update':
        # Handle critical error alerts
        error_data = {
            'error_message': status_data.get('error_message', ''),
            'level': status_data.get('level', 'error'),
            'timestamp': status_data.get('timestamp', ''),
            'system_context': status_data.get('system_context', {})
        }
        
        # Emit error alert to frontend
        socketio.emit('error_update', error_data)
        
        # Log error to backend
        print(f"[{error_data['timestamp']}] [ERROR] {error_data['error_message']}")
        
        # Update system state with error
        system_state['error_message'] = error_data['error_message']
        system_state['last_error_time'] = error_data['timestamp']
        
    elif callback_type == 'exception_details':
        # Handle detailed exception information
        exception_data = {
            'error_details': status_data.get('error_details', {}),
            'timestamp': status_data.get('timestamp', '')
        }
        
        # Emit detailed exception to frontend (for debugging)
        socketio.emit('exception_details', exception_data)
        
        # Log exception details to backend
        details = exception_data['error_details']
        print(f"[{exception_data['timestamp']}] [EXCEPTION] {details.get('operation', 'Unknown')}: {details.get('exception_type', '')}")
        print(f"  Message: {details.get('exception_message', '')}")
        
    elif callback_type == 'system_stopped':
        # Handle system stop with performance summary
        performance_summary = status_data.get('performance_summary', {})
        
        # Emit final performance summary
        socketio.emit('system_stopped', {
            'performance_summary': performance_summary,
            'timestamp': status_data.get('timestamp', '')
        })
        
        # Update system state
        system_state['running'] = False
        system_state['performance'].update(performance_summary)
        
        # Log comprehensive stop summary
        print(f"[SYSTEM STOP] Final throughput: {performance_summary.get('images_per_minute', 0):.1f} img/min")
        
    elif callback_type == 'shutter_sound':
        # Handle camera shutter sound feedback
        shutter_data = {
            'image_number': status_data.get('image_number', 0),
            'timestamp': status_data.get('timestamp', '')
        }
        
        # Emit shutter sound event to frontend
        socketio.emit('shutter_sound', shutter_data)
        
        # Log shutter event (optional, for debugging)
        print(f"[SHUTTER] 📸 Image {shutter_data['image_number']} captured")
        
    elif callback_type == 'performance_update':
        # Handle real-time performance updates with health metrics
        socketio.emit('performance_update', {
            'performance': status_data.get('performance', {}),
            'image_count': status_data.get('performance', {}).get('total_images', 0),
            'health': status_data.get('health', {}),
            'timestamp': status_data.get('timestamp', datetime.now().isoformat())
        })

    else:
        # Legacy status update handling
        system_state.update(status_data)

        # Emit real-time performance updates
        if 'performance' in status_data:
            socketio.emit('performance_update', {
                'performance': status_data['performance'],
                'image_count': status_data.get('image_count', 0),
                'health': status_data.get('health', {}),
                'timestamp': datetime.now().isoformat()
            })

        # Legacy logging
        message = status_data.get('message', 'Status update')
        print(f"Legacy status: {message}")

def get_camera_system():
    """Get optimized camera system instance with error recovery"""
    global camera_system
    if camera_system is None and not SIMULATION_MODE:
        try:
            camera_system = get_system_instance(status_callback)
        except Exception as e:
            print(f"❌ Failed to initialize camera system: {e}")
            print("⚠️ System will run in limited mode - check camera hardware and connections")
            import traceback
            traceback.print_exc()
            # Don't raise - allow server to start even if camera fails
            return None
    return camera_system

def generate_enhanced_camera_stream():
    """Optimized camera streaming using camera system's existing stream buffer"""
    global streaming, connected_clients
    print("Starting optimized camera stream using existing camera buffer...")
    
    frames_sent = 0
    last_frame_time = 0
    target_fps = 15  # Reduced target for stability
    frame_interval = 1.0 / target_fps
    
    while streaming and connected_clients > 0:
        try:
            frame_start = time.time()
            
            if not SIMULATION_MODE:
                system = get_camera_system()
                if system and hasattr(system, 'get_stream_frame'):
                    # Use camera system's existing stream buffer (no additional camera access)
                    frame = system.get_stream_frame()
                    if frame is not None:
                        # Only emit if enough time has passed (frame rate control)
                        current_time = time.time()
                        if current_time - last_frame_time >= frame_interval:
                            try:
                                # Use moderate quality to reduce bandwidth
                                success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                                if success:
                                    frame_base64 = base64.b64encode(buffer).decode('utf-8')
                                    
                                    socketio.emit('frame_update', {
                                        'frame': frame_base64,
                                        'resolution': f"{frame.shape[1]}x{frame.shape[0]}",
                                        'quality': 75,
                                        'timestamp': current_time,
                                        'frame_count': frames_sent
                                    })
                                    
                                    frames_sent += 1
                                    last_frame_time = current_time
                                    
                                    # Debug output every 30 frames
                                    if frames_sent % 30 == 0:
                                        print(f"Web video: {frames_sent} frames sent to {connected_clients} clients")
                                        
                            except Exception as emit_error:
                                print(f"Error emitting frame: {emit_error}")
                        else:
                            # Skip frame to maintain timing
                            pass
                    else:
                        # No frame available, brief pause
                        time.sleep(0.05)
                else:
                    print("Camera system not available - waiting...")
                    time.sleep(0.5)
            else:
                print("System in simulation mode - no video streaming")
                time.sleep(1)
            
            # Frame rate control
            frame_time = time.time() - frame_start
            if frame_time < frame_interval:
                time.sleep(frame_interval - frame_time)
                
        except Exception as e:
            print(f"Web streaming error: {e}")
            time.sleep(0.1)  # Error recovery
    
    print(f"Web camera stream ended. Total frames sent: {frames_sent}")

@socketio.on('connect')
def handle_connect(auth=None):
    """Handle enhanced client connection with error recovery"""
    global streaming, stream_thread, connected_clients

    try:
        print(f"Client connected to enhanced stream (total clients: {connected_clients + 1})")
        connected_clients += 1

        # Send connection confirmation with system info
        emit('connection_confirmed', {
            'timestamp': datetime.now().isoformat(),
            'simulation_mode': SIMULATION_MODE,
            'system_ready': not SIMULATION_MODE,  # Don't initialize camera here
            'streaming_active': streaming,
            'connected_clients': connected_clients
        })
    except Exception as e:
        print(f"Error in connect handler: {e}")
        import traceback
        traceback.print_exc()
    
    # Force start streaming if not already active
    with stream_lock:
        if not streaming and not SIMULATION_MODE:
            streaming = True
            stream_thread = threading.Thread(target=generate_enhanced_camera_stream, daemon=True)
            stream_thread.start()
            print("Started enhanced streaming thread for web interface")
            
            # Send immediate test frame
            system = get_camera_system()
            if system:
                test_frame = system.get_stream_frame()
                if test_frame is not None:
                    success, buffer = cv2.imencode('.jpg', test_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    if success:
                        frame_base64 = base64.b64encode(buffer).decode('utf-8')
                        emit('frame_update', {
                            'frame': frame_base64,
                            'resolution': f"{test_frame.shape[1]}x{test_frame.shape[0]}",
                            'quality': 90,
                            'timestamp': time.time(),
                            'frame_count': 0
                        })
                        print("Sent test frame to client")
    
    # Send initial system status
    try:
        system = get_camera_system()
        if system:
            status = system.get_status()
            emit('system_status_update', status)
    except Exception as e:
        print(f"Error sending initial status: {e}")

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    global streaming, connected_clients
    
    print("Client disconnected from stream")
    connected_clients = max(0, connected_clients - 1)
    
    with stream_lock:
        if connected_clients == 0:
            streaming = False

@socketio.on('request_system_info')
def handle_system_info_request():
    """Handle request for detailed system information"""
    system = get_camera_system()
    system_info = {
        'simulation_mode': SIMULATION_MODE,
        'camera_available': system is not None,
        'system_status': system.get_status() if system else None,
        'performance_metrics': system_state.get('performance', {}),
        'timestamp': datetime.now().isoformat()
    }
    emit('system_info_response', system_info)

@socketio.on('request_log_history')
def handle_log_history_request():
    """Send recent log history to newly connected client"""
    # This could be enhanced to maintain a log history buffer
    emit('log_history', {
        'logs': [],  # Could maintain recent logs in memory
        'message': 'Log history feature ready',
        'timestamp': datetime.now().isoformat()
    })

@socketio.on('request_performance_metrics')
def handle_performance_request():
    """Send current performance metrics"""
    system = get_camera_system()
    if system and hasattr(system, 'get_status'):
        status = system.get_status()
        performance_data = {
            'current_metrics': status.get('performance', {}),
            'system_status': {
                'running': system_state.get('running', False),
                'image_count': system_state.get('image_count', 0),
                'throughput_target': '25+ images/minute'
            },
            'timestamp': datetime.now().isoformat()
        }
    else:
        performance_data = {
            'current_metrics': system_state.get('performance', {}),
            'system_status': system_state,
            'timestamp': datetime.now().isoformat()
        }
    
    emit('performance_metrics', performance_data)

@socketio.on('clear_error_alerts')
def handle_clear_errors():
    """Clear error alerts on frontend request"""
    global system_state
    system_state['error_message'] = None
    system_state['last_error_time'] = None
    emit('errors_cleared', {'timestamp': datetime.now().isoformat()})

@socketio.on('request_queue_status')
def handle_queue_status_request():
    """Send current queue status for monitoring"""
    system = get_camera_system()
    if system and hasattr(system, 'upload_queue'):
        queue_status = {
            'upload_queue_size': system.upload_queue.qsize() if hasattr(system, 'upload_queue') else 0,
            'webhook_queue_size': system.webhook_queue.qsize() if hasattr(system, 'webhook_queue') else 0,
            'max_queue_size': system.__class__.__dict__.get('QUEUE_MAX_SIZE', 5),
            'queue_full_events': getattr(system, 'queue_full_events', 0),
            'dropped_frames': getattr(system, 'dropped_frames', 0),
            'timestamp': datetime.now().isoformat()
        }
    else:
        queue_status = {
            'upload_queue_size': 0,
            'webhook_queue_size': 0,
            'message': 'System not available',
            'timestamp': datetime.now().isoformat()
        }
    
    emit('queue_status', queue_status)

@app.route('/')
def index():
    """Serve React frontend"""
    dist_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frontend', 'dist')
    return send_from_directory(dist_dir, 'index.html')

@app.route('/diagnostics')
def diagnostics():
    """Serve diagnostics page for troubleshooting (vanilla HTML)"""
    return render_template('diagnostics.html')

# Serve React static assets (CSS, JS, images)
@app.route('/assets/<path:path>')
def serve_react_assets(path):
    """Serve React build assets"""
    assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frontend', 'dist', 'assets')
    return send_from_directory(assets_dir, path)

# Serve React favicon and other static files
@app.route('/<path:filename>')
def serve_react_static(filename):
    """Serve React static files (favicon, robots.txt, etc.)"""
    dist_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frontend', 'dist')
    static_files = ['favicon.ico', 'robots.txt', 'placeholder.svg']
    if filename in static_files:
        file_path = os.path.join(dist_dir, filename)
        if os.path.exists(file_path):
            return send_from_directory(dist_dir, filename)
    # If not a static file and not an API route, return index.html for React Router
    if not filename.startswith('api/') and not filename.startswith('socket.io/'):
        return send_from_directory(dist_dir, 'index.html')

@app.route('/api/camera/feed')
def camera_feed():
    """Enhanced HTTP camera feed with higher quality"""
    if SIMULATION_MODE:
        return jsonify({'error': 'Camera feed not available in simulation mode'}), 404
    
    def generate_hq_frames():
        system = get_camera_system()
        while True:
            try:
                if system and hasattr(system, 'get_stream_frame'):
                    frame = system.get_stream_frame()
                    if frame is not None:
                        success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        if success:
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(0.04)  # 25 FPS
            except Exception as e:
                print(f"Error generating HQ frame: {e}")
                time.sleep(0.1)
    
    try:
        return Response(generate_hq_frames(),
                       mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        print(f"Error in camera feed: {e}")
        return jsonify({'error': 'Failed to get camera feed'}), 500

@app.route('/api/status')
def get_status():
    """Get enhanced system status with performance metrics"""
    if SIMULATION_MODE:
        return jsonify({**system_state, 'simulation_mode': True})
    else:
        system = get_camera_system()
        if system:
            status = system.get_status()
            system_state.update(status)
        return jsonify({**system_state, 'simulation_mode': False})

@app.route('/api/performance')
def get_performance_metrics():
    """Get detailed performance metrics"""
    system = get_camera_system()
    performance_data = {
        'timestamp': datetime.now().isoformat(),
        'system_metrics': system.get_status().get('performance', {}) if system else {},
        'streaming_metrics': {
            'connected_clients': connected_clients,
            'streaming_active': streaming,
            'stream_resolution': '1280x720',
            'stream_quality': 95
        }
    }
    return jsonify(performance_data)

@app.route('/api/start', methods=['POST'])
def start_system():
    """Start the enhanced OCR processing system with comprehensive error handling"""
    try:
        data = request.get_json()
        service_provider = data.get('service_provider')

        if not service_provider:
            return jsonify({'error': 'Service provider is required'}), 400

        if system_state['running']:
            return jsonify({'error': 'System is already running'}), 400

        if SIMULATION_MODE:
            return jsonify({'error': 'Cannot start real system in simulation mode'}), 400

        try:
            system = get_camera_system()
            if not system:
                return jsonify({
                    'error': 'Camera system not available',
                    'details': 'Hardware initialization failed. Check camera and ESP32 connections.',
                    'troubleshooting': [
                        'Verify camera cable is connected',
                        'Check ESP32 is connected via USB',
                        'Ensure no other processes are using the camera',
                        'Check system logs for detailed error messages'
                    ]
                }), 500

            run_id = system.start_system(service_provider)

            # Update global state
            system_state['running'] = True
            system_state['service_provider'] = service_provider
            system_state['current_run_id'] = run_id
            system_state['start_time'] = datetime.now().isoformat()

            # Emit system start event to all connected clients
            socketio.emit('system_started', {
                'service_provider': service_provider,
                'run_id': run_id,
                'timestamp': datetime.now().isoformat()
            })

            print(f"✅ System started successfully: {service_provider} (Run ID: {run_id})")

            return jsonify({
                'message': f'Enhanced OCR system started with {service_provider}',
                'run_id': run_id,
                'service_provider': service_provider,
                'ocr_optimized': True,
                'status': 'running'
            })

        except Exception as start_error:
            print(f"❌ System start error: {start_error}")
            import traceback
            traceback.print_exc()

            # Attempt recovery
            system_state['running'] = False
            system_state['error_message'] = str(start_error)

            return jsonify({
                'error': f'Failed to start system: {str(start_error)}',
                'details': 'System startup failed. Check hardware connections and system logs.',
                'recovery': 'Ensure camera and ESP32 are properly connected, then try again.'
            }), 500

    except Exception as e:
        print(f"❌ API /start error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': f'Server error: {str(e)}',
            'details': 'Internal server error during startup request'
        }), 500

@app.route('/api/stop', methods=['POST'])
def stop_system():
    """Stop the system with performance summary"""
    try:
        system = get_camera_system()
        final_count = system_state['image_count']
        
        if not SIMULATION_MODE and system:
            performance_summary = system.get_status().get('performance', {})
            system.stop_system()
            
            # Emit system stop event with performance data
            socketio.emit('system_stopped', {
                'final_image_count': final_count,
                'performance_summary': performance_summary,
                'timestamp': datetime.now().isoformat()
            })

        return jsonify({
            'message': 'Enhanced system stopped successfully',
            'final_image_count': final_count,
            'performance_summary': performance_summary if not SIMULATION_MODE else {}
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/pause', methods=['POST'])
def pause_system():
    """Pause the system"""
    if not system_state['running']:
        return jsonify({'error': 'System is not running'}), 400

    try:
        if not SIMULATION_MODE:
            system = get_camera_system()
            if system:
                system.pause_system()
                
        socketio.emit('system_paused', {'timestamp': datetime.now().isoformat()})
        return jsonify({'message': 'System paused'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/resume', methods=['POST'])
def resume_system():
    """Resume the system"""
    if not system_state['running']:
        return jsonify({'error': 'System is not running'}), 400

    try:
        if not SIMULATION_MODE:
            system = get_camera_system()
            if system:
                system.resume_system()
                
        socketio.emit('system_resumed', {'timestamp': datetime.now().isoformat()})
        return jsonify({'message': 'System resumed'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/capture-test', methods=['POST'])
def capture_test_image():
    """Manual high-quality test capture"""
    if SIMULATION_MODE:
        return jsonify({'error': 'Cannot capture real images in simulation mode'}), 400

    try:
        system = get_camera_system()
        if not system:
            return jsonify({'error': 'Camera system not available'}), 500

        # Set up test parameters
        if not system.service_provider:
            system.service_provider = "TEST"
        if not system.run_id:
            system.run_id = "test_" + str(uuid.uuid4())[:8]

        success = system.capture_and_process_optimized(999, "manual_test")

        if success:
            return jsonify({
                'message': 'High-quality test image captured and processed',
                'run_id': system.run_id,
                'optimized_for_ocr': True
            })
        else:
            return jsonify({'error': 'Failed to capture test image'}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats')
def get_stats():
    """Get enhanced system statistics"""
    uptime = "00:00:00"
    if system_state['start_time']:
        try:
            start = datetime.fromisoformat(system_state['start_time'])
            now = datetime.now()
            delta = now - start
            uptime = str(delta).split('.')[0]
        except:
            pass

    stats = {
        'total_runs': system_state['total_runs'],
        'current_run_images': system_state['image_count'],
        'current_service_provider': system_state['service_provider'],
        'uptime': uptime,
        'last_activity': system_state['last_image_time'],
        'simulation_mode': SIMULATION_MODE,
        'performance': system_state.get('performance', {}),
        'streaming': {
            'connected_clients': connected_clients,
            'active': streaming,
            'quality': 95
        }
    }
    
    return jsonify(stats)

@app.route('/api/health')
def health_check():
    """Enhanced health check with system diagnostics"""
    system = get_camera_system()
    health_data = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'simulation_mode': SIMULATION_MODE,
        'system_running': system_state['running'],
        'camera_available': system is not None,
        'streaming_active': streaming,
        'connected_clients': connected_clients
    }
    
    # Lightweight connectivity check to Node-RED webhook route
    node_red_status = {
        'url': NODE_RED_URL,
        'reachable': False,
        'status_code': None,
        'latency_ms': None
    }
    try:
        start = time.perf_counter()
        resp = requests.head(NODE_RED_URL, timeout=1)
        node_red_status['status_code'] = resp.status_code
        node_red_status['reachable'] = True
        node_red_status['latency_ms'] = int((time.perf_counter() - start) * 1000)
    except requests.RequestException:
        # Leave defaults indicating not reachable
        pass
    health_data['node_red'] = node_red_status

    if system:
        health_data['camera_status'] = 'operational'
        health_data['performance_metrics'] = system.get_status().get('performance', {})
    else:
        health_data['camera_status'] = 'unavailable'
    
    return jsonify(health_data)

# Keep existing API endpoints for backwards compatibility
@app.route('/api/settings', methods=['GET', 'POST'])
def settings():
    """Get or update system settings"""
    if request.method == 'GET':
        return jsonify({
            'aws_region': 'eu-north-1',
            'bucket_name': 'ocrstorage4d',
            # Backwards-compatible key, now reflecting Node-RED endpoint instead of n8n
            'webhook_url': NODE_RED_URL,
            # Explicit key for Node-RED to avoid confusion in clients
            'node_red_url': NODE_RED_URL,
            'simulation_mode': SIMULATION_MODE,
            'ocr_optimized': True,
            'capture_resolution': '2560x1440',
            'capture_quality': 95,
            'stream_resolution': '1280x720',
            'stream_quality': 95
        })
    else:
        data = request.get_json()
        return jsonify({'message': 'Settings updated'})

def cleanup_on_exit():
    """Enhanced cleanup function"""
    print("Shutting down enhanced system...")
    global streaming, connected_clients
    streaming = False
    connected_clients = 0
    
    if not SIMULATION_MODE:
        cleanup_system()

def cleanup_all():
    """Complete enhanced cleanup"""
    global streaming, stream_thread, connected_clients
    streaming = False
    connected_clients = 0
    
    # Wait for stream thread to finish
    if stream_thread and stream_thread.is_alive():
        stream_thread.join(timeout=3)
    
    cleanup_on_exit()

# Register cleanup function
atexit.register(cleanup_all)

def start_health_monitor():
    """Background health monitoring thread"""
    def health_monitor():
        while True:
            try:
                time.sleep(60)  # Check every minute
                status = {
                    'running': system_state.get('running', False),
                    'connected_clients': connected_clients,
                    'streaming': streaming,
                    'timestamp': datetime.now().isoformat()
                }
                print(f"[HEALTH] System status: Running={status['running']}, Clients={status['connected_clients']}, Streaming={status['streaming']}")
            except Exception as e:
                print(f"[HEALTH] Monitor error: {e}")

    monitor_thread = threading.Thread(target=health_monitor, daemon=True)
    monitor_thread.start()
    print("[HEALTH] Health monitor started")

def start_status_emitter():
    """Emit system status at regular intervals."""
    def status_emitter():
        while True:
            try:
                system = get_camera_system()
                if system:
                    status = system.get_status()
                    socketio.emit('status', status)
            except Exception as e:
                print(f"[STATUS_EMITTER] Error: {e}")
            socketio.sleep(1) 

    socketio.start_background_task(target=status_emitter)

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')

    mode_text = "SIMULATION MODE" if SIMULATION_MODE else "OCR-OPTIMIZED HARDWARE MODE"
    print("=" * 80)
    print(f"🚀 Starting Enhanced Flask server in {mode_text}...")
    print("=" * 80)
    print("Features:")
    print("  ✓ High-res OCR processing")
    print("  ✓ Enhanced web streaming")
    print("  ✓ Performance monitoring")
    print("  ✓ Real-time Socket.IO analytics")
    print("  ✓ Auto-recovery error handling")
    print("=" * 80)
    print(f"📡 Access the web interface at: http://192.168.1.15:5000")
    print(f"📊 API health check: http://192.168.1.15:5000/api/health")
    print("=" * 80)

    # Start health monitor
    start_health_monitor()
    start_status_emitter()

    # Pre-initialize camera system to catch errors early
    if not SIMULATION_MODE:
        print("🔧 Pre-initializing camera system...")
        try:
            cam_sys = get_camera_system()
            if cam_sys:
                print("✅ Camera system initialized successfully")
            else:
                print("⚠️ Camera system initialization failed - server will run with limited functionality")
        except Exception as e:
            print(f"⚠️ Camera pre-init error: {e}")
            print("⚠️ Server will start anyway - check hardware connections")

    try:
        print("\n🌐 Starting SocketIO server...")
        socketio.run(
            app,
            debug=False,
            host='0.0.0.0',
            port=5000,
            allow_unsafe_werkzeug=True
        )
    except KeyboardInterrupt:
        print("\n\n⚠️ Keyboard interrupt received - shutting down gracefully...")
        cleanup_all()
    except Exception as e:
        print(f"\n\n❌ Server error: {e}")
        import traceback
        traceback.print_exc()
        print("\n🔄 Attempting cleanup...")
        cleanup_all()