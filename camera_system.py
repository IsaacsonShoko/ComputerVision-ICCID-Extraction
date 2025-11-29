# camera_system.py - Optimized for OCR processing with dual-stream architecture
import warnings
import os
import sys
from picamera2 import Picamera2
import cv2
import time
import requests
import base64
import json
from datetime import datetime
import uuid
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from time import sleep
import threading
import numpy as np
import atexit
import signal
import queue
import concurrent.futures
from functools import wraps
import logging
import serial
import serial.tools.list_ports
import glob
import random

# Edge Impulse imports for SIM card detection
try:
    from edge_impulse_linux.image import ImageImpulseRunner
    EDGE_IMPULSE_AVAILABLE = True
    print("✅ Edge Impulse Linux SDK loaded successfully")
except ImportError as e:
    EDGE_IMPULSE_AVAILABLE = False
    print(f"⚠️ Edge Impulse Linux SDK not available: {e}")
    print("📦 Detection features disabled - system will capture all frames")

# AWS S3 Configuration - LOADED FROM ENVIRONMENT VARIABLES FOR SECURITY
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "eu-north-1")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "ocrstorage4d")

# ESP32 Configuration
ESP32_PORT = '/dev/ttyUSB0'  # Corrected for Raspberry Pi
BAUD_RATE = 115200
esp32_ser = None

# AWS IoT Core Configuration (Primary messaging - replaces webhooks)
IOT_ENDPOINT = os.getenv("AWS_IOT_ENDPOINT", "")  # e.g., "xxxxx-ats.iot.eu-north-1.amazonaws.com"
IOT_CERT_PATH = os.getenv("AWS_IOT_CERT", "certs/device.cert.pem")
IOT_KEY_PATH = os.getenv("AWS_IOT_KEY", "certs/device.private.key")
IOT_CA_PATH = os.getenv("AWS_IOT_CA", "certs/AmazonRootCA1.pem")
IOT_CLIENT_ID = os.getenv("AWS_IOT_CLIENT_ID", "raspberrypi-alpha")
IOT_TOPIC = os.getenv("AWS_IOT_TOPIC", "pi-imaging/detections")

# AWS IoT MQTT Publisher
class AWSIoTPublisher:
    """AWS IoT Core MQTT publisher for reliable message delivery"""

    def __init__(self):
        self.mqtt_connection = None
        self.connected = False
        self.connection_lock = threading.Lock()

    def connect(self):
        """Establish MQTT connection to AWS IoT Core"""
        if not IOT_ENDPOINT:
            print("⚠️ AWS IoT endpoint not configured - MQTT disabled")
            return False

        try:
            from awscrt import mqtt
            from awsiot import mqtt_connection_builder

            # Build MQTT connection with X.509 certificates
            self.mqtt_connection = mqtt_connection_builder.mtls_from_path(
                endpoint=IOT_ENDPOINT,
                cert_filepath=IOT_CERT_PATH,
                pri_key_filepath=IOT_KEY_PATH,
                ca_filepath=IOT_CA_PATH,
                client_id=IOT_CLIENT_ID,
                clean_session=False,
                keep_alive_secs=30,
                on_connection_interrupted=self._on_connection_interrupted,
                on_connection_resumed=self._on_connection_resumed
            )

            # Connect
            connect_future = self.mqtt_connection.connect()
            connect_future.result(timeout=10)

            self.connected = True
            print(f"✅ AWS IoT MQTT connected to {IOT_ENDPOINT}")
            return True

        except ImportError:
            print("⚠️ AWS IoT SDK not installed. Run: pip install awsiotsdk")
            return False
        except Exception as e:
            print(f"❌ AWS IoT connection failed: {e}")
            return False

    def _on_connection_interrupted(self, connection, error, **kwargs):
        print(f"⚠️ AWS IoT connection interrupted: {error}")
        self.connected = False

    def _on_connection_resumed(self, connection, return_code, session_present, **kwargs):
        print(f"✅ AWS IoT connection resumed (code: {return_code})")
        self.connected = True

    def publish(self, payload, topic=None):
        """Publish message to AWS IoT Core with QoS 1 (at least once)"""
        if not self.mqtt_connection or not self.connected:
            return False

        try:
            from awscrt import mqtt

            topic = topic or IOT_TOPIC
            message = json.dumps(payload)

            # Publish with QoS 1 for guaranteed delivery
            publish_future, packet_id = self.mqtt_connection.publish(
                topic=topic,
                payload=message,
                qos=mqtt.QoS.AT_LEAST_ONCE
            )

            # Wait for acknowledgment (with timeout)
            publish_future.result(timeout=5)
            return True

        except Exception as e:
            print(f"❌ MQTT publish failed: {e}")
            return False

    def disconnect(self):
        """Gracefully disconnect from AWS IoT Core"""
        if self.mqtt_connection:
            try:
                disconnect_future = self.mqtt_connection.disconnect()
                disconnect_future.result(timeout=5)
                print("✅ AWS IoT MQTT disconnected")
            except Exception as e:
                print(f"⚠️ MQTT disconnect error: {e}")
            finally:
                self.connected = False

# Global IoT publisher instance (always initialized - IoT is the primary messaging)
iot_publisher = AWSIoTPublisher()

# Optimized settings for better performance - PROVEN WORKING SPEED FROM TEST
CONVEYOR_STEPS = 1568  # 16cm movement for 9-card layout
CAPTURE_QUALITY = 95  # Higher quality for OCR
STREAM_QUALITY = 85   # Moderate quality for web
CAPTURE_TIMEOUT = 1.0  # 1 second timeout for image capture operations
DETECTION_SETTLE_DELAY = 3.0  # Pause after detection before moving belt (reduces motion blur / false positives)
PROCESSING_DELAY = 1.0  # Brief pause between belt stabilization and next cycle (prevents immediate jump to next image)
STEPPER_DELAY = 0.001  # PROVEN WORKING: 1ms from standalone test - drives belt smoothly
BELT_STABILIZATION_TIME = 1.0  # 1 second for belt to stabilize after movement (increased for better stability)
MAX_UPLOAD_WORKERS = 4  # Increased concurrent uploads
QUEUE_MAX_SIZE = 8  # Increased queue size for better buffering

# Persistent retry configuration
FAILED_UPLOADS_DIR = "failed_uploads"    # Directory for failed upload persistence
FAILED_IOT_MESSAGES_DIR = "failed_iot_messages"  # Directory for failed IoT messages
CIRCUIT_BREAKER_THRESHOLD = 5            # Failures before opening circuit
CIRCUIT_BREAKER_TIMEOUT = 60             # Seconds to wait before retrying after circuit opens

# Stream Configuration
STREAM_WIDTH = 1024      # Stream resolution width (720p for smooth streaming)
STREAM_HEIGHT = 1024      # Stream resolution height (720p for smooth streaming)
# Note:  provides good quality with smooth performance for detection model
# Lower resolution than capture (2560x1440) but much faster streaming

# Detection Configuration
ML_WIDTH = 320           # ML model requirement for detection
ML_HEIGHT = 320          # ML model requirement for detection
DETECTION_CONFIDENCE_THRESHOLD = 0.3  # Minimum confidence for SIM card detection (lowered for new multi-SIM model)
DETECTION_NMS_THRESHOLD = 0.4         # IoU threshold for Non-Maximum Suppression
ML_DETECTION_TIMEOUT = 3.0  # Allow up to 3.0s for ML detection (increased to handle multiple detections with debug output)
CROP_SIZE = 96           # Standard crop size for individual SIM card images (96x96)

# Suppress warnings
warnings.filterwarnings("ignore")

# Global variables for GPIO device management
_gpio_devices = {}
_gpio_lock = threading.Lock()

def safe_gpio_device(pin_number, device_name):
    """Safely create or reuse GPIO device"""
    global _gpio_devices, _gpio_lock

    with _gpio_lock:
        if device_name in _gpio_devices:
            return _gpio_devices[device_name]

        try:
            device = OutputDevice(pin_number, initial_value=False)
            _gpio_devices[device_name] = device
            print(f"GPIO device {device_name} initialized on pin {pin_number}")
            return device
        except Exception as e:
            print(f"Error creating GPIO device {device_name} on pin {pin_number}: {e}")
            try:
                import lgpio
                handle = lgpio.gpiochip_open(0)
                try:
                    lgpio.gpio_free(handle, pin_number)
                except:
                    pass
                lgpio.gpiochip_close(handle)

                device = OutputDevice(pin_number, initial_value=False)
                _gpio_devices[device_name] = device
                print(f"GPIO device {device_name} created after cleanup")
                return device
            except Exception as e2:
                print(f"Failed to create GPIO device {device_name}: {e2}")
                raise e

def cleanup_all_gpio():
    """Cleanup all GPIO devices"""
    global _gpio_devices, _gpio_lock

    with _gpio_lock:
        for device_name, device in _gpio_devices.items():
            try:
                if device:
                    device.close()
                    print(f"Closed GPIO device: {device_name}")
            except Exception as e:
                print(f"Error closing GPIO device {device_name}: {e}")

        _gpio_devices.clear()

        try:
            import lgpio
            handle = lgpio.gpiochip_open(0)
            pins_to_free = [13, 19, 21, 20, 16]
            for pin in pins_to_free:
                try:
                    lgpio.gpio_free(handle, pin)
                except:
                    pass
            lgpio.gpiochip_close(handle)
            print("GPIO cleanup completed")
        except Exception as e:
            print(f"GPIO cleanup warning: {e}")


def send_command(cmd, wait_time=1, retries=2):
    """
    ROBUST serial communication with automatic retry

    Features:
    - Clears buffer before sending (prevents stale data)
    - Waits for "OK" confirmation
    - Auto-retry on failure
    - Detailed error reporting
    """
    if not esp32_ser:
        print("❌ ESP32 not connected.")
        return False

    for attempt in range(retries + 1):
        # Clear any stale data in buffer
        if esp32_ser.in_waiting:
            stale = esp32_ser.read_all()
            if attempt == 0:  # Only log on first attempt
                print(f"🗑️  Cleared {len(stale)} stale bytes")

        # Send command
        if attempt > 0:
            print(f"🔄 Retry {attempt}/{retries}: {cmd}")
        else:
            print(f"↑ Sending: {cmd}")

        esp32_ser.write(f"{cmd}\n".encode())

        # Wait for response with timeout
        response = ""
        deadline = time.time() + wait_time + 0.5  # Extra buffer for safety

        while time.time() < deadline:
            if esp32_ser.in_waiting:
                line = esp32_ser.readline().decode('utf-8', errors='ignore').strip()
                response += line + " "
                if "OK" in line:
                    print(f"← Response: {response.strip()}")
                    return True
            time.sleep(0.01)

        # If we get here, command failed
        if response:
            print(f"⚠️  Got response but no 'OK': {response.strip()}")
        else:
            print(f"⚠️  No response from ESP32")

        # Small delay before retry
        if attempt < retries:
            time.sleep(0.3)

    # All retries exhausted
    print(f"❌ Command failed after {retries + 1} attempts: {cmd}")
    return False


def connect_esp32():
    """Connect to ESP32 with improved error handling."""
    global esp32_ser
    try:
        print(f"Attempting to connect to ESP32 on {ESP32_PORT}...")
        esp32_ser = serial.Serial(ESP32_PORT, BAUD_RATE, timeout=2)
        time.sleep(2)  # Wait for the connection to establish
        if esp32_ser.in_waiting:
            msg = esp32_ser.readline().decode('utf-8', errors='ignore').strip()
            print(f"ESP32 initial message: {msg}")
        print("✅ ESP32 connected successfully.")
        return True
    except serial.SerialException as e:
        print(f"❌ Serial Error: {e}")
        print("   Please check:")
        print("   1. Is the ESP32 connected to the correct USB port?")
        print(f"   2. Is the port set to '{ESP32_PORT}' in the script?")
        print("   3. Is another program (like Arduino IDE or another script) using the port?")
        return False
    except Exception as e:
        print(f"❌ An unexpected error occurred during ESP32 connection: {e}")
        return False


class DetectionProcessor:
    """SIM Card Detection Processor using Edge Impulse SSD model for multi-card detection"""
    
    def __init__(self):
        self.detection_runner = None
        self.detection_labels = []
        self.detection_input_width = 0
        self.detection_input_height = 0
        self.initialized = False
        self.detection_lock = threading.Lock()  # Protect detection runner from concurrent access

        # Detection filtering parameters (MATCHING OCR_Test.py exactly!)
        self.confidence_threshold = 0.3  # RESTORED: Same as OCR_Test.py for consistent performance
        self.nms_threshold = DETECTION_NMS_THRESHOLD
        self.debug_mode = True  # ENABLED: See all detections to troubleshoot
        self.force_rgb_mode = True
        
        # Size filtering parameters (reject boxes too large to be SIM cards)
        # SIM card is ~25mm x 15mm, at 1024x1024 resolution should be ~200x120 pixels max
        # TIGHTENED: More restrictive to eliminate false positives on empty belt
        self.max_bbox_width_ratio = 0.25   # Max 25% of frame width (REDUCED from 35%)
        self.max_bbox_height_ratio = 0.25  # Max 25% of frame height (REDUCED from 35%)
        self.min_bbox_width_ratio = 0.06   # Min 6% of frame width (INCREASED from 5%)
        self.min_bbox_height_ratio = 0.06  # Min 6% of frame height (INCREASED from 5%)
        self.min_bbox_area_ratio = 0.015   # Min 1.5% of frame area (INCREASED from 1%)
        self.max_bbox_area_ratio = 0.12    # Max 12% of frame area (REDUCED from 20%)
        
    def initialize_models(self):
        """Initialize Edge Impulse detection model"""
        if not EDGE_IMPULSE_AVAILABLE:
            print("⚠️ Edge Impulse not available - detection disabled")
            return False
            
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            detection_model_path = os.path.join(script_dir, "models", "simcard_detection.eim")
            
            if not os.path.exists(detection_model_path):
                print(f"⚠️ SIM detection model not found: {detection_model_path}")
                print("📦 System will capture all frames without detection filtering")
                return False
            
            print(f"📁 Loading detection model: {os.path.basename(detection_model_path)}")
            self.detection_runner = ImageImpulseRunner(detection_model_path)
            detection_model_info = self.detection_runner.init()
            
            print(f'✅ Detection model loaded: "{detection_model_info["project"]["owner"]} / {detection_model_info["project"]["name"]}"')
            self.detection_labels = detection_model_info['model_parameters']['labels']
            self.detection_input_width = detection_model_info['model_parameters']['image_input_width']
            self.detection_input_height = detection_model_info['model_parameters']['image_input_height']
            print(f"🏷️ Detection labels: {self.detection_labels}")
            print(f"📐 Detection input size: {self.detection_input_width}x{self.detection_input_height}")
            print(f"🎯 Confidence threshold: {self.confidence_threshold}")
            
            self.initialized = True
            return True
            
        except Exception as e:
            print(f"❌ Model initialization failed: {e}")
            print("📦 System will capture all frames without detection filtering")
            return False
    
    def is_valid_simcard_size(self, bbox, frame_width, frame_height):
        """Validate that bounding box size is reasonable for a SIM card"""
        width = bbox['width']
        height = bbox['height']
        
        # Calculate ratios
        width_ratio = width / frame_width
        height_ratio = height / frame_height
        
        # Check if too small
        if width_ratio < self.min_bbox_width_ratio or height_ratio < self.min_bbox_height_ratio:
            if self.debug_mode:
                print(f"🚫 Size Filter: Box too small ({width}x{height}, {width_ratio:.1%}x{height_ratio:.1%}) - likely noise")
            return False
        
        # Check if too large (likely false positive - full frame detection)
        if width_ratio > self.max_bbox_width_ratio or height_ratio > self.max_bbox_height_ratio:
            if self.debug_mode:
                print(f"🚫 Size Filter: Box too large ({width}x{height}, {width_ratio:.1%}x{height_ratio:.1%} of frame) - rejecting")
            return False
        
        area_ratio = (width * height) / (frame_width * frame_height)
        if area_ratio < self.min_bbox_area_ratio or area_ratio > self.max_bbox_area_ratio:
            if self.debug_mode:
                print(f"🚫 Size Filter: Area ratio {area_ratio:.1%} out of bounds")
            return False
        
        # Valid size
        if self.debug_mode:
            print(f"✅ Size Filter: Valid SIM card size ({width}x{height}, {width_ratio:.1%}x{height_ratio:.1%} of frame)")
        return True
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) of two bounding boxes"""
        x1_1, y1_1, w1, h1 = box1['x'], box1['y'], box1['width'], box1['height']
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        
        x1_2, y1_2, w2, h2 = box2['x'], box2['y'], box2['width'], box2['height']
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2
        
        # Calculate intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def apply_nms(self, detections, frame_width=ML_WIDTH, frame_height=ML_HEIGHT):
        """Apply Non-Maximum Suppression to filter duplicate detections - MATCHING OCR_Test.py"""
        if not detections:
            return []
        
        # Sort by confidence (highest first)
        detections.sort(key=lambda x: x['value'], reverse=True)
        
        filtered_detections = []
        
        for current in detections:
            # Skip if confidence too low
            if current['value'] < self.confidence_threshold:
                if self.debug_mode:
                    print(f"🔍 DEBUG: Skipping low confidence detection: {current['label']} = {current['value']:.3f} (threshold: {self.confidence_threshold})")
                continue
            
            # Skip if size is unreasonable for a SIM card
            if not self.is_valid_simcard_size(current, frame_width, frame_height):
                continue
            
            # Check if this detection overlaps significantly with any already accepted detection
            should_keep = True
            for accepted in filtered_detections:
                iou = self.calculate_iou(current, accepted)
                if iou > self.nms_threshold:
                    # If same class, keep the higher confidence one (already sorted)
                    if current['label'] == accepted['label']:
                        if self.debug_mode:
                            print(f"🔄 NMS: Duplicate {current['label']} removed (IoU: {iou:.2f})")
                        should_keep = False
                        break
                    # If different classes but high overlap, keep both but note the conflict
                    else:
                        if self.debug_mode:
                            print(f"⚠️ Class conflict: {current['label']} vs {accepted['label']} (IoU: {iou:.2f})")
            
            if should_keep:
                filtered_detections.append(current)
                if self.debug_mode:
                    print(f"✅ Kept detection: {current['label']} ({current['value']:.2f})")
        
        if self.debug_mode:
            print(f"📊 NMS Results: {len(detections)} → {len(filtered_detections)} detections")
        
        return filtered_detections
    
    def process_frame(self, frame):
        """
        Process frame for SIM card detection
        Returns: List of detections with bounding boxes and confidence scores
        """
        if not self.initialized:
            return []
        
        results = []
        
        try:
            # Prepare ML-sized frame for detection
            ml_frame = cv2.resize(frame, (ML_WIDTH, ML_HEIGHT), interpolation=cv2.INTER_AREA)
            
            # Debug: Check frame formats (matching OCR_Test.py)
            if self.debug_mode:
                print(f"🔍 Input frame shape: {frame.shape}")
                print(f"🔍 ML frame shape: {ml_frame.shape}")
                print(f"🔍 Frame dtype: {ml_frame.dtype}")
                print(f"🔍 Running SIM card detection on {ML_WIDTH}x{ML_HEIGHT} frame...")
            
            # Handle color space for Edge Impulse (models expect RGB input)
            # PiCamera2 "main" stream is already in RGB888 format - use directly!
            # This matches OCR_Test.py exactly
            if hasattr(self, 'force_rgb_mode') and self.force_rgb_mode:
                # PiCamera2 is in RGB mode - use frame directly (correct!)
                rgb_frame = ml_frame
            elif len(frame.shape) == 3 and frame.shape[2] == 3:
                # If input is BGR (e.g., from OpenCV imread), convert to RGB
                # But PiCamera2 gives us RGB, so this branch shouldn't execute
                rgb_frame = cv2.cvtColor(ml_frame, cv2.COLOR_BGR2RGB)
            else:
                # Fallback - use as-is
                rgb_frame = ml_frame

            # Use lock to prevent concurrent access to detection runner (fixes Edge Impulse message ID conflicts)
            with self.detection_lock:
                detection_features, processed_img = self.detection_runner.get_features_from_image(rgb_frame)
                detection_result = self.detection_runner.classify(detection_features)

            # Get raw detections
            raw_detections = detection_result.get("result", {}).get("bounding_boxes", [])
            
            # Debug: Show all raw detections before filtering
            if self.debug_mode:
                dsp_time = detection_result['timing']['dsp']
                classification_time = detection_result['timing']['classification']
                # Ensure timing values are numbers, not strings
                if isinstance(dsp_time, str):
                    dsp_time = float(dsp_time)
                if isinstance(classification_time, str):
                    classification_time = float(classification_time)
                processing_time = dsp_time + classification_time
                print(f'📊 Detection: Found {len(raw_detections)} bounding boxes ({processing_time} ms.)')
                print(f"🔍 DEBUG: Raw detections count: {len(raw_detections)}")
                for i, bbox in enumerate(raw_detections):
                    print(f"   Detection {i+1}: {bbox['label']} confidence={bbox['value']:.3f} x={bbox['x']} y={bbox['y']} w={bbox['width']} h={bbox['height']}")
            
            # Apply NMS filtering with size validation (using ML resolution for size checks)
            filtered_detections = self.apply_nms(raw_detections, ML_WIDTH, ML_HEIGHT)
            
            # Debug: Show filtered results
            if self.debug_mode and filtered_detections:
                print(f"🔍 DEBUG: After NMS filtering: {len(filtered_detections)} detections remain")
                for det in filtered_detections:
                    print(f"   Final: {det['label']} confidence={det['value']:.3f}")
            elif self.debug_mode:
                print(f"🔍 DEBUG: No detections after filtering (confidence threshold: {self.confidence_threshold})")
            
            # Process each filtered SIM card detection with coordinate scaling
            for bbox in filtered_detections:
                # Scale coordinates from ML resolution to original frame resolution
                scaled_bbox = self.scale_coordinates_to_display(bbox, ml_frame.shape, frame.shape)
                
                sim_detection = {
                    'label': bbox['label'],
                    'confidence': bbox['value'],
                    'bbox': scaled_bbox
                }
                
                if self.debug_mode:
                    print(f'📋 Detection: {bbox["label"]} (confidence: {bbox["value"]:.2f})')
                    print(f'📐 Coordinates: ML({bbox["x"]},{bbox["y"]},{bbox["width"]},{bbox["height"]}) → Display({scaled_bbox["x"]},{scaled_bbox["y"]},{scaled_bbox["width"]},{scaled_bbox["height"]})')
                
                results.append(sim_detection)
                    
        except Exception as e:
            import traceback
            print(f"❌ Detection processing error: {e}")
            print(f"❌ Full traceback:")
            traceback.print_exc()

            # Check if Edge Impulse runner crashed and needs reinitialization
            error_msg = str(e).lower()
            if "no data or corrupted data" in error_msg or "wrong id" in error_msg or "connection" in error_msg:
                print(f"⚠️ Detection runner appears crashed. Attempting reinitialization...")
                try:
                    # Cleanup old runner
                    if self.detection_runner:
                        try:
                            self.detection_runner.stop()
                        except:
                            pass

                    # Reinitialize
                    from edge_impulse_linux.image import ImageImpulseRunner
                    self.detection_runner = ImageImpulseRunner(DETECTION_MODEL_FILE)
                    model_info = self.detection_runner.init()
                    print(f"✅ Detection runner reinitialized successfully")
                    print(f"   Model: {model_info['project']['name']}")
                except Exception as reinit_error:
                    print(f"❌ Failed to reinitialize detection runner: {reinit_error}")
                    self.initialized = False

        return results
    
    def scale_coordinates_to_display(self, ml_bbox, ml_shape, display_shape):
        """Scale bounding box coordinates from ML resolution to display resolution"""
        ml_height, ml_width = ml_shape[:2]
        display_height, display_width = display_shape[:2]
        
        # Calculate scaling factors
        scale_x = display_width / ml_width
        scale_y = display_height / ml_height
        
        # Scale coordinates
        scaled_bbox = {
            'x': int(ml_bbox['x'] * scale_x),
            'y': int(ml_bbox['y'] * scale_y),
            'width': int(ml_bbox['width'] * scale_x),
            'height': int(ml_bbox['height'] * scale_y)
        }
        
        if self.debug_mode:
            print(f"🔄 Coordinate scaling: {scale_x:.2f}x, {scale_y:.2f}y")
            print(f"   ML: ({ml_bbox['x']},{ml_bbox['y']},{ml_bbox['width']},{ml_bbox['height']})")
            print(f"   Display: ({scaled_bbox['x']},{scaled_bbox['y']},{scaled_bbox['width']},{scaled_bbox['height']})")
        
        return scaled_bbox
    
    def cleanup(self):
        """Clean shutdown of models"""
        if self.detection_runner:
            try:
                self.detection_runner.stop()
                print("✅ Detection model stopped")
            except Exception as e:
                print(f"⚠️ Error stopping detection model: {e}")

class OptimizedCameraConveyorSystem:
    def _dispense_card_batch(self, forward=True):
        """
        Dispense a batch of 9 SIM cards (3 positions x 3 cards).
        Oscillates direction: forward (0->90->180) or reverse (180->90->0)
        """
        start_time = time.time()

        if forward:
            # Forward direction: 0° -> 90° -> 180°
            positions = [0, 90, 180]
        else:
            # Reverse direction: 180° -> 90° -> 0°
            positions = [180, 90, 0]

        for i, pos in enumerate(positions):
            print(f"Dispensing 3 cards from position {pos}°...")
            send_command(f"POSITION {pos}", 0.5)
            send_command("PUSH 90", 0.4)
            send_command("PUSH 0", 0.4)

        # Stay at last position (no return to home - oscillation pattern)
        dispense_time = time.time() - start_time
        with self.stats_lock:
            self.dispensed_sim_cards_count += 9
            self.dispense_times.append(dispense_time)

    def run_automated_cycle(self):
        """
        Run the automated dispensing and imaging cycle continuously until stopped.

        Sequence: dispense -> move stepper -> capture -> dispense (reversed) -> move stepper -> capture -> repeat
        Pattern: 0->90->180 -> MOVE -> 180->90->0 -> MOVE -> 0->90->180 -> MOVE -> ...
        """
        if not connect_esp32():
            self.addLogEntry("Failed to connect to ESP32 - cycle aborted", 'error')
            return

        # Track dispense direction for oscillation
        dispense_forward = True
        cycle_count = 0

        self.addLogEntry("Starting automated cycle", 'info')

        while self.running and not self.should_stop:
            cycle_count += 1
            self.addLogEntry(f"Cycle {cycle_count}: Dispensing cards ({'forward' if dispense_forward else 'reverse'})", 'info')

            # Step 1: Dispense cards (alternating direction)
            self._dispense_card_batch(forward=dispense_forward)

            # Step 2: Move conveyor belt
            self.addLogEntry(f"Cycle {cycle_count}: Moving conveyor belt", 'info')
            send_command(f"MOVE {CONVEYOR_STEPS} 1", 3.5)

            # Wait for belt to stop and settle before capturing (prevent jitter/blur)
            # CONVEYOR_STEPS=1568 at typical speed takes ~3s to complete + 0.5s settling
            time.sleep(3.5)
            self.addLogEntry(f"Cycle {cycle_count}: Belt settled, ready to capture", 'info')

            # Step 3: Capture and process image
            self.addLogEntry(f"Cycle {cycle_count}: Capturing and processing image", 'info')
            success = self.capture_and_process_optimized(cycle_count)

            if success:
                self.addLogEntry(f"Cycle {cycle_count}: Successfully processed frame", 'info')
            else:
                self.addLogEntry(f"Cycle {cycle_count}: No detections in this frame", 'warning')

            # Toggle direction for next cycle
            dispense_forward = not dispense_forward

        self.addLogEntry(f"Automated cycle stopped after {cycle_count} cycles", 'info')

    def __init__(self, status_callback=None):
        self.camera = None
        self.s3_client = None
        self.running = False
        self.paused = False
        self.should_stop = False
        self.run_id = None
        self.service_provider = None
        self.image_count = 0  # Now counts individual SIM cards uploaded, not frames
        self.frame_count = 0  # Track frames processed
        self.status_callback = status_callback
        self.capture_thread = None
        self.last_image_path = None
        
        # Detection system
        self.detection_processor = DetectionProcessor()
        self.detection_enabled = False
        self.detection_count = 0
        self.last_detection_time = None
        self.last_detections = []  # Store last detections for visual overlay
        self.detection_stats = {
            'total_detections': 0,
            'total_simcards_uploaded': 0,
            'frames_with_detections': 0,
            'frames_without_detections': 0,
            'frames_processed': 0,
            'avg_confidence': 0.0,
            'avg_simcards_per_frame': 0.0
        }
        self.dispensed_sim_cards_count = 0
        self.dispense_times = []
        
        # Dual-stream management
        self.stream_buffer = None
        self.stream_lock = threading.Lock()
        self.stream_active = False
        self.stream_thread = None
        
        # Performance tracking
        self.capture_times = []
        self.capture_timestamps = []
        self.upload_times = []
        self.total_processing_time = 0
        
        # Async processing components
        self.upload_queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)
        self.iot_queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)
        self.upload_executor = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_UPLOAD_WORKERS, thread_name_prefix='S3Upload')
        self.image_count_lock = threading.Lock()
        self.stats_lock = threading.Lock()

        # Performance monitoring
        self.dropped_frames = 0
        self.queue_full_events = 0
        self.upload_errors = 0
        self.iot_errors = 0

        # Circuit breaker for IoT failures
        self.iot_circuit_breaker = {
            'state': 'closed',  # closed, open, half_open
            'failures': 0,
            'last_failure_time': None,
            'last_success_time': None
        }

        # Worker threads list for graceful shutdown
        self.worker_threads = []
        self.shutdown_event = threading.Event()

        # ESP32 serial connection
        self.esp32 = None
        self.esp32_lock = threading.Lock()

        # Initialize persistent directories
        self.init_persistent_dirs()

        # Initialize components
        self.init_esp32_connection()
        self.init_s3_client()
        self.init_camera_optimized()
        self.init_detection_system()

        # Retry any failed webhooks/uploads from previous sessions
        self.retry_failed_operations()

    def init_esp32_connection(self):
        """Initialize serial connection to ESP32 motor controller"""
        try:
            # Try to find ESP32 automatically
            esp32_port = None
            ports = serial.tools.list_ports.comports()
            
            for port in ports:
                # Common ESP32 port patterns on Raspberry Pi
                if 'USB' in port.device or 'ACM' in port.device:
                    try:
                        test_serial = serial.Serial(port.device, 115200, timeout=2)
                        time.sleep(2)  # Wait for ESP32 to initialize
                        if test_serial.in_waiting:
                            response = test_serial.readline().decode('utf-8').strip()
                            if 'READY' in response:
                                esp32_port = port.device
                                test_serial.close()
                                break
                        test_serial.close()
                    except:
                        continue
            
            # If not found automatically, try default port
            if not esp32_port:
                esp32_port = '/dev/ttyUSB0'
            
            self.esp32 = serial.Serial(esp32_port, 115200, timeout=2)
            time.sleep(2)  # Wait for ESP32 initialization
            
            # Clear any startup messages
            while self.esp32.in_waiting:
                self.esp32.readline()
            
            print(f"✅ ESP32 connected on {esp32_port}")
            self.addLogEntry(f"ESP32 motor controller connected on {esp32_port}", 'info')
            
        except Exception as e:
            print(f"❌ Error connecting to ESP32: {e}")
            print("⚠️ Make sure ESP32 is connected via USB")
            raise e
    
    def send_esp32_command(self, command, wait_for_ok=True):
        """Send command to ESP32 and optionally wait for OK response"""
        with self.esp32_lock:
            try:
                self.esp32.write(f"{command}\n".encode('utf-8'))
                
                if wait_for_ok:
                    response = self.esp32.readline().decode('utf-8').strip()
                    if response == "OK":
                        return True
                    else:
                        print(f"⚠️ Unexpected ESP32 response: {response}")
                        return False
                return True
                
            except Exception as e:
                print(f"❌ ESP32 command error: {e}")
                return False

    def init_s3_client(self):
        """Initialize S3 client with optimized settings"""
        try:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                region_name=AWS_REGION,
                config=boto3.session.Config(
                    retries={'max_attempts': 3, 'mode': 'adaptive'},
                    max_pool_connections=10
                )
            )
            print(f"S3 client initialized - Region: {AWS_REGION}")
        except Exception as e:
            print(f"Error initializing S3 client: {e}")
            raise e

    def _cleanup_camera_resources(self):
        """Aggressive camera resource cleanup to handle all edge cases"""
        print("Performing aggressive camera cleanup...")
        
        # Stop any existing streaming
        if hasattr(self, 'stream_active'):
            self.stream_active = False
        
        # Stop and close existing camera with multiple attempts
        if hasattr(self, 'camera') and self.camera:
            for attempt in range(3):
                try:
                    print(f"Camera cleanup attempt {attempt + 1}")
                    if hasattr(self.camera, 'stop'):
                        self.camera.stop()
                    if hasattr(self.camera, 'close'):
                        self.camera.close()
                    break
                except Exception as e:
                    print(f"Camera cleanup attempt {attempt + 1} failed: {e}")
                    if attempt < 2:
                        time.sleep(1)
        
        # Clear camera reference
        self.camera = None
        
        # Try to force release any lingering camera processes
        try:
            import subprocess
            subprocess.run(['sudo', 'pkill', '-f', 'libcamera'], check=False, capture_output=True)
            time.sleep(1)
        except:
            pass
        
        print("Camera cleanup complete")

    def _force_camera_release(self):
        """Emergency camera release for system startup"""
        print("Forcing camera release for fresh start...")
        try:
            # Kill any lingering camera processes
            import subprocess
            result = subprocess.run(['sudo', 'fuser', '/dev/video0'], 
                                  capture_output=True, text=True, check=False)
            if result.returncode == 0 and result.stdout.strip():
                print(f"Found processes using camera: {result.stdout.strip()}")
                subprocess.run(['sudo', 'fuser', '-k', '/dev/video0'], 
                             check=False, capture_output=True)
                time.sleep(2)
            
            # Additional cleanup
            subprocess.run(['sudo', 'pkill', '-f', 'libcamera'], 
                          check=False, capture_output=True)
            time.sleep(1)
        except Exception as e:
            print(f"Force camera release warning: {e}")

    def init_camera_optimized(self):
        """Initialize camera with optimized dual-stream configuration for OCR"""
        try:
            print("Initializing optimized camera system...")
            
            # EMERGENCY: Force camera release first
            self._force_camera_release()
            
            # AGGRESSIVE cleanup - handle all camera resource conflicts
            self._cleanup_camera_resources()
            
            # Reset all stream state
            self.stream_buffer = None
            self.stream_active = False
            
            # Force garbage collection to free camera resources
            import gc
            gc.collect()
            time.sleep(2)  # Give system time to release camera
            
            print("Creating Picamera2 instance...")
            self.camera = Picamera2()
            time.sleep(1)
            
            print("Configuring optimized dual-stream setup...")
            # Optimized configuration for OCR processing
            config = self.camera.create_preview_configuration(
                main={
                    "size": (1024, 1024),  # High resolution for OCR and detection
                    "format": "RGB888"     # Direct RGB for processing
                },
                lores={
                    "size": (STREAM_WIDTH, STREAM_HEIGHT),  # Configurable stream resolution
                    "format": "YUV420"     # Required YUV format
                },
                buffer_count=3  # Balanced buffer count
            )
            
            self.camera.configure(config)
            
            print("Starting camera...")
            self.camera.start()
            time.sleep(1)  # Initial startup
            
            # Set optimized controls for OV5647 IR camera (matching OCR_Test.py settings)
            controls = {
                # Exposure settings for IR lighting
                "ExposureTime": 20000,  # 20ms exposure for dim IR lighting
                "AnalogueGain": 6.0,    # Higher gain for dim IR sensitivity
                
                # Contrast and brightness for IR imaging
                "Contrast": 2.2,        # INCREASED: Higher contrast helps yellow cards stand out
                "Brightness": 0.3,      # INCREASED: Better illumination for yellow cards
                "Saturation": 0.8,      # INCREASED: Better color differentiation (yellow vs background)
                
                # Focus and sharpness
                "Sharpness": 2.0,       # Enhanced sharpness for text clarity
                
                # White balance (important for IR)
                "AwbEnable": False,     # Disable auto white balance for IR
                "ColourGains": (1.4, 1.2),  # TUNED: Adjusted for better yellow card visibility
                
                # Noise reduction
                "NoiseReductionMode": 2,  # Moderate noise reduction
            }
            
            try:
                self.camera.set_controls(controls)
                print("✅ OV5647 IR camera controls applied successfully")
                print(f"📊 IR Optimized Settings:")
                print(f"   Exposure: {controls['ExposureTime']}μs, Gain: {controls['AnalogueGain']}x")
                print(f"   Contrast: {controls['Contrast']}, Brightness: {controls['Brightness']}")
                print(f"   Sharpness: {controls['Sharpness']}, Saturation: {controls['Saturation']}")
            except Exception as ctrl_e:
                print(f"⚠️ Some camera controls may not be supported: {ctrl_e}")
                print("📹 Camera will use default settings")
            
            time.sleep(1)  # Camera stabilization after settings
            
            # Verify both streams work
            print("Testing camera streams...")
            main_frame = self.camera.capture_array("main")
            lores_frame = self.camera.capture_array("lores")
            
            if main_frame is None or lores_frame is None:
                raise RuntimeError("Camera stream test failed")
            
            print(f"Camera streams verified:")
            print(f"  Main (OCR): {main_frame.shape}")
            print(f"  Lores (Web): {lores_frame.shape}")
            
            # Initialize stream buffer with first frame
            if lores_frame is not None:
                try:
                    # Convert YUV to BGR for initial frame
                    if len(lores_frame.shape) == 2:  # YUV420 format
                        h, w = 1024, 1024
                        yuv = lores_frame[:h+h//2, :w]
                        initial_bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV420p2BGR)
                    else:
                        initial_bgr = lores_frame
                    
                    # Resize if needed
                    if initial_bgr.shape[0] > 1024:
                        initial_bgr = cv2.resize(initial_bgr, (1024, 1024), interpolation=cv2.INTER_LINEAR)
                    
                    with self.stream_lock:
                        self.stream_buffer = initial_bgr
                    print("Stream buffer initialized with first frame")
                except Exception as e:
                    print(f"Warning: Could not initialize stream buffer: {e}")
            
            # Start optimized streaming thread
            self.stream_active = True
            self.stream_thread = threading.Thread(target=self._optimized_stream_thread, daemon=True)
            self.stream_thread.start()
            
            print("Optimized camera system ready for OCR processing")
            return True
            
        except Exception as e:
            print(f"Camera initialization failed: {e}")
            if hasattr(self, 'camera') and self.camera:
                try:
                    self.camera.close()
                except:
                    pass
            self.camera = None
            self.stream_active = False
            raise e

    def _optimized_stream_thread(self):
        """High-performance streaming thread with 20+ FPS target and improved error handling"""
        print("Starting high-performance stream thread...")
        consecutive_errors = 0
        max_errors = 10
        frame_count = 0
        fps_start = time.time()
        skip_frames = 0
        target_fps = 25
        
        # Wait for camera to be fully initialized
        initialization_timeout = 10  # seconds
        start_wait = time.time()
        while self.stream_active and not self.camera and (time.time() - start_wait) < initialization_timeout:
            print("Waiting for camera initialization...")
            time.sleep(0.5)
        
        if not self.camera:
            print("Camera not available after initialization timeout")
            self.stream_active = False
            return
        
        # Give camera a moment to fully settle
        time.sleep(1)
        
        while self.stream_active:
            try:
                frame_start = time.time()
                
                if self.camera:
                    # Skip frames under heavy load
                    if skip_frames > 0:
                        skip_frames -= 1
                        time.sleep(0.04)  # Frame interval for ~25fps
                        continue
                    
                    try:
                        # Safe frame capture with proper error handling
                        frame = self.camera.capture_array("lores")
                    except Exception as capture_error:
                        print(f"Frame capture error: {capture_error}")
                        consecutive_errors += 1
                        if consecutive_errors >= max_errors:
                            print("Too many capture errors, stopping stream")
                            break
                        time.sleep(0.1)
                        continue
                        
                    if frame is not None:
                        try:
                            # Thread-safe frame processing
                            with self.stream_lock:
                                # Optimized YUV to BGR conversion
                                if len(frame.shape) == 2:  # YUV420 format
                                    h, w = STREAM_HEIGHT, STREAM_WIDTH
                                    yuv = frame[:h+h//2, :w]  # Crop to expected size
                                    bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV420p2BGR)
                                else:
                                    bgr = frame  # Already in correct format
                                    
                                # Optional: resize for better streaming performance
                                if bgr.shape[0] != STREAM_HEIGHT or bgr.shape[1] != STREAM_WIDTH:
                                    bgr = cv2.resize(bgr, (STREAM_WIDTH, STREAM_HEIGHT), interpolation=cv2.INTER_LINEAR)
                                
                                # Store processed frame
                                self.stream_buffer = bgr.copy()
                                consecutive_errors = 0
                            
                            # RE-ENABLED: Stream detection for competition video proof
                            # Reduced frequency (every 30th frame vs every 5th) to minimize load
                            # Detection lock prevents runner crashes from concurrent access
                            if self.detection_enabled and frame_count % 30 == 0:
                                try:
                                    hires_frame = self.camera.capture_array("main")
                                    if hires_frame is not None:
                                        detections = self.detection_processor.process_frame(hires_frame)
                                        self.last_detections = detections if detections else []
                                except Exception as det_error:
                                    # Don't let detection errors crash the stream
                                    pass
                                
                            # FPS monitoring and load balancing
                            frame_count += 1
                            if frame_count % 30 == 0:
                                current_fps = 30 / (time.time() - fps_start)
                                fps_start = time.time()
                                
                                # Adaptive performance management
                                if current_fps < target_fps * 0.8:
                                    skip_frames = 2
                                    if hasattr(self, 'addLogEntry'):
                                        self.addLogEntry(f"Stream performance: {current_fps:.1f}fps (adaptive frame skipping)", 'info')
                                elif current_fps > target_fps * 1.2 and skip_frames == 0:
                                    # Running too fast, add small delay
                                    time.sleep(0.01)
                                    
                        except Exception as conv_error:
                            print(f"Frame processing error: {conv_error}")
                            consecutive_errors += 1
                            
                    else:
                        consecutive_errors += 1
                        if consecutive_errors > 5:
                            print(f"Multiple None frames ({consecutive_errors}), camera may be busy")
                            time.sleep(0.2)  # Longer pause for recovery
                        else:
                            time.sleep(0.05)
                else:
                    print("Camera became unavailable during streaming")
                    time.sleep(0.5)
                    continue
                    
                # Frame timing control with error-aware delays
                frame_time = time.time() - frame_start
                target_time = 1.0 / target_fps
                
                if frame_time < target_time and consecutive_errors == 0:
                    time.sleep(target_time - frame_time)
                elif consecutive_errors > 0:
                    # Slower timing when errors occur
                    time.sleep(0.1)
                    
            except Exception as e:
                consecutive_errors += 1
                print(f"Stream thread error ({consecutive_errors}/{max_errors}): {e}")
                if consecutive_errors >= max_errors:
                    print("Max stream errors reached, stopping")
                    self.stream_active = False
                    break
                time.sleep(0.1)  # Error recovery delay
        
        print("High-performance stream thread ended gracefully")

    def get_stream_frame(self):
        """Get latest high-quality frame for web streaming with detection overlays"""
        try:
            with self.stream_lock:
                if self.stream_buffer is not None:
                    frame = self.stream_buffer.copy()
                    
                    # Draw detection overlays if available
                    if hasattr(self, 'last_detections') and self.last_detections:
                        frame = self.draw_detection_overlays(frame, self.last_detections)
                    
                    # Convert RGB to BGR for JPEG encoding (cv2.imencode expects BGR)
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    return frame_bgr
                
                # If no stream buffer available, try to capture a fallback frame
                if self.camera and hasattr(self.camera, 'capture_array'):
                    try:
                        fallback_frame = self.camera.capture_array("lores")
                        if fallback_frame is not None:
                            # Convert YUV to BGR
                            if len(fallback_frame.shape) == 2:
                                h, w = STREAM_HEIGHT, STREAM_WIDTH
                                yuv = fallback_frame[:h+h//2, :w]
                                bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV420p2BGR)
                            else:
                                bgr = fallback_frame
                            
                            if bgr.shape[0] != STREAM_HEIGHT or bgr.shape[1] != STREAM_WIDTH:
                                bgr = cv2.resize(bgr, (STREAM_WIDTH, STREAM_HEIGHT), interpolation=cv2.INTER_LINEAR)
                            
                            return bgr
                    except Exception as fallback_error:
                        print(f"Fallback frame capture failed: {fallback_error}")
                
                return None
        except Exception as e:
            print(f"Error getting stream frame: {e}")
            return None
    
    def draw_detection_overlays(self, frame, detections):
        """Draw bounding boxes and labels on frame for visual feedback"""
        try:
            # Get frame dimensions for scaling
            frame_h, frame_w = frame.shape[:2]
            
            # Scale factors from high-res (2560x1440) to stream resolution
            scale_x = frame_w / 1024
            scale_y = frame_h / 1024

            for idx, detection in enumerate(detections):
                # Extract bbox and confidence from detection structure
                bbox = detection.get('bbox', {})
                conf = detection.get('confidence', 0.0)
                
                # Scale bounding box coordinates
                x = int(bbox.get('x', 0) * scale_x)
                y = int(bbox.get('y', 0) * scale_y)
                w = int(bbox.get('width', 0) * scale_x)
                h = int(bbox.get('height', 0) * scale_y)
                
                # Skip invalid boxes
                if w <= 0 or h <= 0:
                    continue
                
                # All detection boxes in green (consistent, clean look)
                color = (0, 255, 0)  # Green for all detections
                
                # Draw bounding box (thicker for visibility)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
                
                # Draw label background
                label = f"SIM {idx+1}: {conf:.0%}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x, y - label_size[1] - 8), (x + label_size[0] + 4, y), color, -1)
                
                # Draw label text (black on colored background)
                cv2.putText(frame, label, (x + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Draw detection summary in top-left corner
            if detections:
                avg_conf = sum(d.get('confidence', 0.0) for d in detections) / len(detections)
                summary = f"Detected: {len(detections)} SIM cards (Avg: {avg_conf:.0%})"
                summary_size, _ = cv2.getTextSize(summary, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (5, 5), (15 + summary_size[0], 35), (0, 255, 0), -1)
                cv2.putText(frame, summary, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
        except Exception as e:
            print(f"Error drawing detection overlays: {e}")
            import traceback
            traceback.print_exc()
        
        return frame

    def get_ocr_frame(self):
        """Get high-resolution frame optimized for OCR processing"""
        try:
            if self.camera is None:
                print("❌ Camera not initialized - cannot capture OCR frame")
                return None
                
            frame = self.camera.capture_array("main")
            if frame is not None:
                # Camera outputs RGB888 - return as-is for detection!
                # Detection model expects RGB input (Edge Impulse trained on RGB)
                # OpenCV operations that need BGR can convert locally if needed
                return frame  # Keep as RGB!
            return None
        except Exception as e:
            print(f"Error capturing OCR frame: {e}")
            return None

    def reset_servos(self):
        """Reset all servos to home position (0°)"""
        try:
            if self.send_esp32_command("RESET"):
                self.addLogEntry("✅ Servos reset to home position", 'info')
                return True
            return False
        except Exception as e:
            self.addLogEntry(f"❌ Servo reset error: {e}", 'error')
            return False
    
    def dispense_sim_cards(self):
        """Execute automated SIM card dispensing sequence

        CURRENTLY DISABLED: Servo dispensing for manual card placement
        - Servo dispensing disabled - place SIM cards manually on belt
        - Only belt movement is active for positioning cards under camera
        - To re-enable dispenser: Uncomment the DISPENSE commands below

        Dispenses 6 cards total (2 rows × 3 cards):
        - Row 1: Position platform at 0°, push 3 cards, retract
        - Row 2: Position platform at 90°, push 3 cards, retract
        - Move belt 10cm forward to position cards under camera
        """
        try:
            self.addLogEntry("🎯 Manual mode: Place 6 SIM cards on belt by hand", 'info')

            # DISABLED: Servo dispensing - uncomment when dispenser is ready
            # # Dispense from Row 1 (3 cards)
            # if not self.send_esp32_command("DISPENSE 1"):
            #     raise Exception("Row 1 dispense failed")
            # time.sleep(0.5)  # Brief pause for cards to drop

            # # Dispense from Row 2 (3 cards)
            # if not self.send_esp32_command("DISPENSE 2"):
            #     raise Exception("Row 2 dispense failed")
            # time.sleep(0.5)  # Brief pause for cards to drop

            # Move belt 10cm to position cards under camera (ALWAYS ACTIVE)
            self.move_conveyor_belt_optimized(distance_multiplier=1.0)

            self.addLogEntry("✅ Manual dispense mode: Belt moved - place next 6 cards manually", 'info')
            return True

        except Exception as e:
            self.addLogEntry(f"❌ Belt movement error: {e}", 'error')
            return False
    
    def manual_dispense_row(self, row_number):
        """Manually dispense 3 cards from specific row (for testing)
        
        Args:
            row_number: 1 or 2
        """
        try:
            if row_number not in [1, 2]:
                raise ValueError("Row number must be 1 or 2")
            
            if not self.send_esp32_command(f"DISPENSE {row_number}"):
                raise Exception(f"Row {row_number} dispense failed")
            
            self.addLogEntry(f"✅ Dispensed 3 cards from Row {row_number}", 'info')
            return True
            
        except Exception as e:
            self.addLogEntry(f"❌ Manual dispense error: {e}", 'error')
            return False

    def init_detection_system(self):
        """Initialize SIM card detection system"""
        try:
            print("🔍 Initializing SIM card detection system...")
            if self.detection_processor.initialize_models():
                self.detection_enabled = True
                print("✅ Detection system enabled - capturing only when SIM cards detected")
                print("📊 Multi-crop mode: Each detected SIM card will be uploaded separately")
            else:
                self.detection_enabled = False
                print("⚠️ Detection system disabled - capturing all frames")
        except Exception as e:
            print(f"⚠️ Detection initialization warning: {e}")
            self.detection_enabled = False
            print("📦 Continuing without detection - capturing all frames")

    def init_persistent_dirs(self):
        """Initialize directories for persistent retry storage"""
        try:
            os.makedirs(FAILED_UPLOADS_DIR, exist_ok=True)
            os.makedirs(FAILED_IOT_MESSAGES_DIR, exist_ok=True)
            print(f"✅ Persistent storage directories initialized")
        except Exception as e:
            print(f"⚠️ Could not create persistent directories: {e}")

    def save_failed_upload(self, upload_task):
        """Save failed upload task to disk for later retry"""
        try:
            filename = f"{FAILED_UPLOADS_DIR}/upload_{upload_task['image_number']}_{uuid.uuid4().hex[:8]}.json"
            # Save metadata and s3_key, but not the actual image bytes (too large)
            task_metadata = {
                's3_key': upload_task['s3_key'],
                'metadata': upload_task['metadata'],
                'image_number': upload_task['image_number'],
                'note': 'Image bytes not persisted - upload cannot be retried automatically'
            }
            with open(filename, 'w') as f:
                json.dump(task_metadata, f, indent=2)
            self.addLogEntry(f"💾 Saved failed upload metadata to disk: {filename}", 'warning')
            return True
        except Exception as e:
            self.addLogEntry(f"❌ Failed to save upload metadata to disk: {e}", 'error')
            return False

    def retry_failed_operations(self):
        """Retry failed IoT messages and uploads from previous sessions"""
        # Retry failed IoT messages
        iot_files = glob.glob(f"{FAILED_IOT_MESSAGES_DIR}/*.json")
        if iot_files:
            self.addLogEntry(f"🔄 Found {len(iot_files)} failed IoT messages to retry", 'info')
            for iot_file in iot_files:
                try:
                    with open(iot_file, 'r') as f:
                        iot_message = json.load(f)

                    # Try to publish to IoT
                    if iot_publisher.connected and iot_publisher.publish(iot_message):
                        os.remove(iot_file)
                        self.addLogEntry(f"✅ Successfully retried IoT message: {iot_file}", 'info')
                    else:
                        self.addLogEntry(f"⚠️ IoT retry failed, will try again later: {iot_file}", 'warning')
                except Exception as e:
                    self.addLogEntry(f"❌ Error retrying IoT message {iot_file}: {e}", 'error')

        # Note: Upload retries are more complex since we don't persist image bytes
        upload_files = glob.glob(f"{FAILED_UPLOADS_DIR}/*.json")
        if upload_files:
            self.addLogEntry(f"ℹ️ Found {len(upload_files)} failed upload records (cannot auto-retry without image data)", 'warning')

    def move_conveyor_belt_optimized(self, distance_multiplier=1.0):
        """Move conveyor belt via ESP32 controller"""
        try:
            # Calculate steps (1568 steps = 16cm)
            steps_to_move = int(CONVEYOR_STEPS * distance_multiplier)

            # Send move command to ESP32 (direction 1 = forward)
            command = f"MOVE {steps_to_move} 1"
            if not self.send_esp32_command(command):
                raise Exception("ESP32 move command failed")

            # Belt stabilization pause - critical for stable imaging
            time.sleep(BELT_STABILIZATION_TIME)

            self.addLogEntry(f"Conveyor moved {16.0 * distance_multiplier:.1f}cm ({steps_to_move} steps)", 'info')
            return True
            
        except Exception as e:
            self.addLogEntry(f"❌ Belt movement error: {e}", 'error')
            return False

    def run_ml_detection_with_timeout(self, image, timeout=ML_DETECTION_TIMEOUT):
        """Run ML detection with configurable timeout for better accuracy"""
        
        def detection_worker():
            nonlocal result
            try:
                # Run detection on captured frame (same as before)
                result = self.detection_processor.process_frame(image)
            except Exception as e:
                result = None
        
        result = None
        detection_thread = threading.Thread(target=detection_worker)
        detection_thread.start()
        detection_thread.join(timeout=timeout)
        
        if detection_thread.is_alive():
            self.addLogEntry(f"⚠️ ML detection timed out after {timeout}s", 'warning')
            return []
        
        return result or []

    def capture_and_process_optimized(self, image_number, capture_type="auto"):
        """Async optimized capture with detection-triggered capture and multi-crop uploads"""
        capture_start = time.time()
        
        try:
            # Capture high-resolution frame with timeout
            ocr_frame = None
            capture_attempts = 0
            max_attempts = int(CAPTURE_TIMEOUT * 10)  # 10 attempts per second
            
            while ocr_frame is None and capture_attempts < max_attempts:
                try:
                    ocr_frame = self.get_ocr_frame()
                    if ocr_frame is not None:
                        break
                except Exception as frame_error:
                    print(f"Frame capture attempt {capture_attempts + 1} failed: {frame_error}")
                
                capture_attempts += 1
                if ocr_frame is None:
                    time.sleep(0.1)
            
            if ocr_frame is None:
                raise ValueError(f"Frame capture failed after {CAPTURE_TIMEOUT}s timeout")
            
            capture_time = time.time() - capture_start
            capture_timestamp = time.time()

            with self.stats_lock:
                self.capture_times.append(capture_time)
                self.capture_timestamps.append(capture_timestamp)
                if len(self.capture_times) > 300:
                    self.capture_times = self.capture_times[-300:]
                if len(self.capture_timestamps) > 300:
                    self.capture_timestamps = self.capture_timestamps[-300:]
            
            # Check for SIM card detection before processing (if detection enabled)
            if self.detection_enabled:
                detection_start = time.time()
                
                # Run detection on captured frame with extended timeout (ocr_frame is already RGB from camera)
                detections = self.run_ml_detection_with_timeout(ocr_frame, ML_DETECTION_TIMEOUT)
                detection_time = time.time() - detection_start
                
                # Store detections for visual overlay
                self.last_detections = detections if detections else []
                
                # Store detections for visual overlay on stream
                self.last_detections = detections if detections else []
                
                # Update detection statistics
                with self.stats_lock:
                    self.detection_stats['frames_processed'] += 1
                    if detections:
                        self.detection_stats['frames_with_detections'] += 1
                        self.detection_stats['total_detections'] += len(detections)
                        confidences = [d['confidence'] for d in detections]
                        self.detection_stats['avg_confidence'] = sum(confidences) / len(confidences)
                        self.detection_count = len(detections)
                        self.last_detection_time = time.time()
                    else:
                        self.detection_stats['frames_without_detections'] += 1
                        self.detection_count = 0
                
                # Check if SIM cards were detected
                print(f"\n{'='*60}")
                print(f"[CAPTURE DEBUG] Detection completed for frame {image_number}")
                print(f"[CAPTURE DEBUG] Detections found: {len(detections) if detections else 0}")
                print(f"[CAPTURE DEBUG] Detection time: {detection_time:.3f}s")
                print(f"[CAPTURE DEBUG] Detections list: {detections}")
                print(f"{'='*60}\n")

                if not detections:
                    self.addLogEntry(f"🔍 Frame {image_number}: No SIM cards detected - waiting ({detection_time:.3f}s)", 'info')
                    return False  # Return False = don't move belt, keep looking

                print(f"[CAPTURE DEBUG] Proceeding to upload section with {len(detections)} detections")
                self.addLogEntry(f"✅ Detection: {len(detections)} SIM card(s) found (avg conf: {self.detection_stats['avg_confidence']:.2f}, {detection_time:.3f}s)", 'info')
                
                # Now crop and upload each detected SIM card individually
                simcards_uploaded = 0
                simcards_failed = 0
                
                timestamp_base = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                
                # Process each detected SIM card
                for detection_idx, detection in enumerate(detections, 1):
                    try:
                        bbox = detection['bbox']
                        x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
                        
                        # Clamp coordinates to frame boundaries
                        frame_height, frame_width = ocr_frame.shape[:2]
                        x = max(0, x)
                        y = max(0, y)
                        w = min(frame_width - x, w)
                        h = min(frame_height - y, h)
                        
                        if w <= 0 or h <= 0:
                            self.addLogEntry(f"⚠️ Invalid crop dimensions for detection {detection_idx}", 'warning')
                            simcards_failed += 1
                            continue
                        
                        # Crop the detected SIM card region (ocr_frame is RGB)
                        cropped_sim = ocr_frame[y:y+h, x:x+w]
                        
                        if cropped_sim.size == 0:
                            self.addLogEntry(f"⚠️ Empty crop for detection {detection_idx}", 'warning')
                            simcards_failed += 1
                            continue
                        
                        # Convert RGB to BGR for OpenCV operations (resize, encode)
                        # OpenCV expects BGR format for image processing
                        cropped_sim_bgr = cv2.cvtColor(cropped_sim, cv2.COLOR_RGB2BGR)
                        
                        # Resize to 96x96 for OCR processing
                        resized_crop = cv2.resize(cropped_sim_bgr, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_CUBIC)
                        
                        # Encode crop to JPEG
                        encode_start = time.time()
                        success, crop_encoded = cv2.imencode('.jpg', resized_crop, [cv2.IMWRITE_JPEG_QUALITY, CAPTURE_QUALITY])
                        if not success:
                            self.addLogEntry(f"⚠️ Failed to encode crop {detection_idx}", 'warning')
                            simcards_failed += 1
                            continue
                        
                        crop_bytes = crop_encoded.tobytes()
                        encode_time = time.time() - encode_start
                        
                        # Generate unique filename for this crop
                        filename = f'crop_{self.service_provider.lower()}_{timestamp_base}_frame{image_number:04d}_sim{detection_idx:02d}.jpg'
                        
                        # Prepare metadata for this specific crop
                        crop_metadata = {
                            'run_id': self.run_id,
                            'service_provider': self.service_provider,
                            'image_name': filename,
                            'image_number': f"{image_number}-{detection_idx}",  # Required for IoT publishing
                            'frame_number': image_number,
                            'detection_number': detection_idx,
                            'total_detections_in_frame': len(detections),
                            'timestamp': datetime.now().isoformat(),
                            'capture_type': capture_type,
                            'crop_resolution': f'{CROP_SIZE}x{CROP_SIZE}',
                            'original_bbox': {'x': x, 'y': y, 'width': w, 'height': h},
                            'quality': CAPTURE_QUALITY,
                            'detection_confidence': detection['confidence'],
                            'detection_label': detection['label'],
                            'capture_time': capture_time,
                            'detection_time': detection_time,
                            'encode_time': encode_time
                        }
                        
                        s3_key = f"simcard-crops/{self.run_id}/{filename}"
                        
                        # Queue this crop for async S3 upload
                        try:
                            upload_task = {
                                'img_bytes': crop_bytes,
                                's3_key': s3_key,
                                'metadata': crop_metadata,
                                'image_number': f"{image_number}-{detection_idx}"
                            }
                            self.upload_queue.put(upload_task, timeout=1)
                            simcards_uploaded += 1
                            
                        except queue.Full:
                            with self.stats_lock:
                                self.queue_full_events += 1
                            self.addLogEntry(f"⚠️ Upload queue full! Dropping SIM card {detection_idx}", 'warning')
                            simcards_failed += 1
                    
                    except Exception as crop_error:
                        self.addLogEntry(f"❌ Error processing SIM card {detection_idx}: {crop_error}", 'error')
                        simcards_failed += 1
                
                # Update image count based on SIM cards uploaded
                with self.image_count_lock:
                    self.image_count += simcards_uploaded
                
                with self.stats_lock:
                    self.detection_stats['total_simcards_uploaded'] += simcards_uploaded
                    if self.detection_stats['frames_processed'] > 0:
                        self.detection_stats['avg_simcards_per_frame'] = self.detection_stats['total_simcards_uploaded'] / self.detection_stats['frames_processed']
                
                total_time = time.time() - capture_start
                with self.stats_lock:
                    self.total_processing_time += total_time
                self.addLogEntry(f"📦 Frame {image_number}: {simcards_uploaded} SIM cards queued, {simcards_failed} failed (total: {total_time:.3f}s)", 'info')
                self.emit_performance_update()
                
                # Send shutter sound feedback with crop count
                if self.status_callback:
                    try:
                        self.status_callback({
                            'type': 'shutter_sound',
                            'image_number': image_number,
                            'timestamp': datetime.now().isoformat(),
                            'detections': len(detections),
                            'simcards_uploaded': simcards_uploaded
                        })
                    except Exception as sound_error:
                        print(f"Shutter sound callback error: {sound_error}")
                
                return simcards_uploaded > 0
            
            else:
                # Detection disabled - fall back to full frame capture
                self.addLogEntry("⚠️ Detection disabled - fallback to full frame capture", 'warning')
                # TODO: Implement fallback full-frame capture if needed
                self.emit_performance_update()
                return False

        except Exception as e:
            with self.stats_lock:
                self.dropped_frames += 1
            self.log_exception(f"Capture processing for frame {image_number}", e, 'error')
            self.emit_performance_update()
            return False

    def send_to_iot(self, s3_key, s3_url, metadata):
        """Send message to AWS IoT Core via MQTT"""
        import traceback

        iot_message = {
            # Standard data
            "s3_key": s3_key,
            "s3_url": s3_url,
            "s3_bucket": S3_BUCKET_NAME,
            "s3_region": AWS_REGION,
            "run_id": metadata['run_id'],
            "image_name": metadata['image_name'],
            "image_number": metadata['image_number'],
            "timestamp": metadata['timestamp'],
            "service_provider": metadata.get('service_provider', 'Unknown'),
            "capture_type": metadata.get('capture_type', 'unknown'),
            "resolution": metadata.get('resolution', '2560x1440'),
            "quality": metadata.get('quality', CAPTURE_QUALITY),
            "device_id": IOT_CLIENT_ID,
        }

        # DEBUG: Log the attempt
        print(f"\n{'='*60}")
        print(f"[IoT DEBUG] Attempting to send to AWS IoT Core")
        print(f"[IoT DEBUG] Topic: {IOT_TOPIC}")
        print(f"[IoT DEBUG] Image number: {metadata['image_number']}")
        print(f"[IoT DEBUG] IoT connected: {iot_publisher.connected}")
        print(f"[IoT DEBUG] IoT endpoint: {IOT_ENDPOINT}")
        print(f"[IoT DEBUG] Message preview: {json.dumps(iot_message, indent=2)[:500]}...")
        print(f"{'='*60}\n")

        # Publish to AWS IoT
        if not iot_publisher.connected:
            print(f"[IoT ERROR] IoT publisher not connected! Saving message for retry.")
            self.addLogEntry(f"⚠️ IoT not connected - saving message for retry", 'warning')
            self.save_failed_iot_message(iot_message)
            return False

        try:
            print(f"[IoT DEBUG] Calling iot_publisher.publish()...")
            publish_result = iot_publisher.publish(iot_message)
            print(f"[IoT DEBUG] Publish result: {publish_result}")

            if publish_result:
                print(f"[IoT SUCCESS] ✓ Published to IoT: image {metadata['image_number']}")
                self.addLogEntry(f"✓ Published to IoT: image {metadata['image_number']}", 'info')
                self.record_iot_success()
                return True
            else:
                print(f"[IoT ERROR] Publish returned False for image {metadata['image_number']}")
                self.addLogEntry(f"❌ IoT publish failed for image {metadata['image_number']}", 'error')
                self.record_iot_failure()
                self.save_failed_iot_message(iot_message)
                return False
        except Exception as e:
            print(f"[IoT EXCEPTION] Error during IoT publish:")
            print(f"[IoT EXCEPTION] Error type: {type(e).__name__}")
            print(f"[IoT EXCEPTION] Error message: {str(e)}")
            print(f"[IoT EXCEPTION] Full traceback:")
            traceback.print_exc()
            self.addLogEntry(f"❌ IoT error: {e}", 'error')
            self.record_iot_failure()
            self.save_failed_iot_message(iot_message)
            return False

    def save_failed_iot_message(self, message_data):
        """Save failed IoT message to disk for later retry"""
        try:
            os.makedirs(FAILED_IOT_MESSAGES_DIR, exist_ok=True)
            filename = f"{FAILED_IOT_MESSAGES_DIR}/iot_{message_data.get('image_number', 'unknown')}_{uuid.uuid4().hex[:8]}.json"
            with open(filename, 'w') as f:
                json.dump(message_data, f, indent=2)
            self.addLogEntry(f"💾 Saved failed IoT message to disk: {filename}", 'info')
        except Exception as e:
            self.addLogEntry(f"❌ Failed to save IoT message to disk: {e}", 'error')

    def record_iot_success(self):
        """Record successful IoT publish"""
        self.iot_circuit_breaker['failures'] = 0
        self.iot_circuit_breaker['last_success_time'] = time.time()
        if self.iot_circuit_breaker['state'] == 'half_open':
            self.iot_circuit_breaker['state'] = 'closed'
            self.addLogEntry("✅ IoT circuit breaker closed (service recovered)", 'info')

    def record_iot_failure(self):
        """Record failed IoT publish and potentially open circuit breaker"""
        self.iot_circuit_breaker['failures'] += 1
        self.iot_circuit_breaker['last_failure_time'] = time.time()
        with self.stats_lock:
            self.iot_errors += 1

        if self.iot_circuit_breaker['failures'] >= CIRCUIT_BREAKER_THRESHOLD:
            if self.iot_circuit_breaker['state'] != 'open':
                self.iot_circuit_breaker['state'] = 'open'
                self.addLogEntry(
                    f"⚠️ IoT circuit breaker opened after {self.iot_circuit_breaker['failures']} consecutive failures. "
                    f"Will retry in {CIRCUIT_BREAKER_TIMEOUT}s",
                    'warning'
                )

    def async_upload_to_s3(self, img_bytes, s3_key, metadata, image_number):
        """Async S3 upload with error handling and retry logic"""
        import random
        max_retries = 3
        for attempt in range(max_retries):
            try:
                upload_start = time.time()

                self.s3_client.put_object(
                    Bucket=S3_BUCKET_NAME,
                    Key=s3_key,
                    Body=img_bytes,
                    ContentType='image/jpeg',
                    Metadata={
                        'run_id': str(metadata.get('run_id', '')),
                        'service_provider': str(metadata.get('service_provider', '')),
                        'image_number': str(metadata.get('image_number', '')),
                        'timestamp': metadata.get('timestamp', ''),
                        'capture_type': str(metadata.get('capture_type', '')),
                        'resolution': str(metadata.get('resolution', '')),
                        'quality': str(metadata.get('quality', '')),
                        'optimized_for': 'ocr_processing'
                    }
                )

                upload_time = time.time() - upload_start
                with self.stats_lock:
                    self.upload_times.append(upload_time)
                    # Keep only last 100 measurements
                    if len(self.upload_times) > 100:
                        self.upload_times = self.upload_times[-100:]

                s3_url = f"https://{S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{s3_key}"
                self.addLogEntry(f"S3 upload completed for image {image_number} ({upload_time:.2f}s)", 'info')

                # Send to AWS IoT Core
                updated_metadata = {**metadata, 'upload_time': upload_time}
                self.send_to_iot(s3_key, s3_url, updated_metadata)

                return True, s3_url

            except Exception as e:
                with self.stats_lock:
                    self.upload_errors += 1
                self.log_exception(f"S3 upload for image {image_number} (attempt {attempt + 1})", e, 'error')
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + random.uniform(0, 1) # Exponential backoff with jitter
                    self.addLogEntry(f"Retrying S3 upload for image {image_number} in {wait_time:.2f}s...", 'warning')
                    time.sleep(wait_time)
                else:
                    self.addLogEntry(f"S3 upload for image {image_number} failed after {max_retries} attempts", 'error')
                    return False, None
        return False, None

    def start_background_workers(self):
        """Start background processing workers"""
        def upload_worker():
            """Background S3 upload worker"""
            while self.running or not self.upload_queue.empty():
                try:
                    upload_task = self.upload_queue.get(timeout=1.0)
                    if upload_task is None:  # Shutdown signal
                        break

                    success, s3_url = self.async_upload_to_s3(
                        upload_task['img_bytes'],
                        upload_task['s3_key'],
                        upload_task['metadata'],
                        upload_task['image_number']
                    )

                    self.upload_queue.task_done()

                except queue.Empty:
                    continue
                except Exception as e:
                    self.log_exception("Upload worker", e, 'error')

        self.worker_threads = []
        # Start worker threads (non-daemon for graceful shutdown)
        for i in range(MAX_UPLOAD_WORKERS):
            thread = threading.Thread(target=upload_worker, name=f'S3Upload-{i}', daemon=False)
            thread.start()
            self.worker_threads.append(thread)

        self.addLogEntry(f"Started {MAX_UPLOAD_WORKERS} S3 upload workers", 'info')

    def stop_background_workers(self):
        """Gracefully stop background workers and wait for queue processing"""
        self.addLogEntry("🛑 Stopping background workers gracefully...", 'info')

        # First, wait for queue to be processed (with timeout)
        queue_size = self.upload_queue.qsize()
        if queue_size > 0:
            self.addLogEntry(f"⏳ Waiting for {queue_size} items in upload queue to be processed...", 'info')
            wait_start = time.time()
            max_wait = 30  # Maximum 30 seconds to process queue

            while not self.upload_queue.empty() and (time.time() - wait_start) < max_wait:
                remaining = self.upload_queue.qsize()
                self.addLogEntry(f"   Queue processing: {remaining} items remaining...", 'info')
                time.sleep(1)

            if self.upload_queue.empty():
                self.addLogEntry("✅ Upload queue processed successfully", 'info')
            else:
                self.addLogEntry(f"⚠️ Timeout: {self.upload_queue.qsize()} items remain in queue", 'warning')

        # Now send shutdown signals to all workers
        self.addLogEntry("Sending shutdown signals to workers...", 'info')
        for _ in range(MAX_UPLOAD_WORKERS):
            try:
                self.upload_queue.put(None, timeout=1)
            except queue.Full:
                self.addLogEntry("Upload queue full during shutdown, forcing shutdown", 'warning')

        # Wait for workers to finish processing
        self.addLogEntry(f"Waiting for {len(self.worker_threads)} worker threads to complete...", 'info')
        for thread in self.worker_threads:
            thread.join(timeout=10)  # Wait up to 10 seconds per thread
            if thread.is_alive():
                self.addLogEntry(f"Warning: Worker thread {thread.name} did not finish in time", 'warning')

        self.addLogEntry("✅ Background workers stopped", 'info')

    def addLogEntry(self, message, level='info'):
        """Enhanced log entry with comprehensive error handling and SocketIO integration"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]  # Include milliseconds
        
        # Add performance context to logs
        perf_info = ""
        if self.capture_times and level == 'info' and ('captured' in message.lower() or 'uploaded' in message.lower()):
            avg_capture = sum(self.capture_times[-5:]) / min(len(self.capture_times), 5)
            avg_upload = sum(self.upload_times[-5:]) / min(len(self.upload_times), 5) if self.upload_times else 0
            perf_info = f" [C:{avg_capture:.2f}s U:{avg_upload:.2f}s]"
        
        # Add queue status for monitoring
        queue_info = ""
        if hasattr(self, 'upload_queue') and hasattr(self, 'webhook_queue'):
            queue_info = f" [Q:U{self.upload_queue.qsize()}/W{self.webhook_queue.qsize()}]"
        
        log_message = f"[{timestamp}]{queue_info} {message}{perf_info}"
        print(log_message)  # Terminal logging
        
        # Enhanced SocketIO logging integration
        if self.status_callback:
            try:
                # Get current performance metrics
                performance_data = {}
                with self.stats_lock:
                    if self.capture_times:
                        performance_data = {
                            'avg_capture_time': sum(self.capture_times[-10:]) / min(len(self.capture_times), 10),
                            'avg_upload_time': sum(self.upload_times[-10:]) / min(len(self.upload_times), 10) if self.upload_times else 0,
                            'images_per_minute': len([ts for ts in self.capture_timestamps if time.time() - ts < 60]) if self.capture_timestamps else 0,
                            'dropped_frames': self.dropped_frames,
                            'queue_full_events': self.queue_full_events,
                            'upload_errors': self.upload_errors,
                            'iot_errors': self.iot_errors
                        }
                
                # Send log update to frontend
                self.status_callback({
                    'type': 'log_update',
                    'log_message': message,
                    'level': level,
                    'timestamp': timestamp,
                    'full_message': log_message,
                    'performance': performance_data,
                    'system_status': {
                        'running': self.running,
                        'paused': self.paused,
                        'image_count': self.image_count,
                        'service_provider': self.service_provider,
                        'run_id': self.run_id
                    }
                })
                
                # Send error alerts for critical issues
                if level in ['error', 'critical']:
                    self.status_callback({
                        'type': 'error_update',
                        'error_message': message,
                        'level': level,
                        'timestamp': timestamp,
                        'system_context': {
                            'running': self.running,
                            'image_count': self.image_count,
                            'service_provider': self.service_provider
                        }
                    })
                    
            except Exception as callback_error:
                print(f"[{timestamp}] LOG CALLBACK ERROR: {callback_error}")

    def log_exception(self, operation_name, exception, level='error'):
        """Specialized exception logging with context"""
        import traceback
        
        error_details = {
            'operation': operation_name,
            'exception_type': type(exception).__name__,
            'exception_message': str(exception),
            'traceback': traceback.format_exc()
        }
        
        error_message = f"{operation_name} failed: {type(exception).__name__}: {str(exception)}"
        self.addLogEntry(error_message, level)
        
        # Send detailed error to frontend
        if self.status_callback:
            try:
                self.status_callback({
                    'type': 'exception_details',
                    'error_details': error_details,
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as callback_error:
                print(f"Exception callback error: {callback_error}")
        
        return error_details

    def start_system(self, service_provider):
        """Start async optimized system with enhanced performance tracking"""
        if self.running:
            raise Exception("System is already running")

        self.service_provider = service_provider
        self.run_id = str(uuid.uuid4())
        
        # Thread-safe initialization
        with self.image_count_lock:
            self.image_count = 0
        
        self.running = True
        self.paused = False
        self.should_stop = False
        self.start_time = datetime.now()
        
        # Reset performance metrics
        with self.stats_lock:
            self.capture_times = []
            self.capture_timestamps = []
            self.upload_times = []
            self.total_processing_time = 0
            self.dropped_frames = 0
            self.queue_full_events = 0
            self.upload_errors = 0
            self.iot_errors = 0
            self.dispensed_sim_cards_count = 0
            self.dispense_times = []

        # Dispenser tracking (dispense every frame in parallel with capture)
        self.dispense_in_progress = False
        self.dispense_lock = threading.Lock()

        self.addLogEntry(f"Starting async OCR system - Provider: {service_provider}", 'info')
        self.addLogEntry(f"Config: 1024x1024@{CAPTURE_QUALITY}%, Stepper: {STEPPER_DELAY*1000:.1f}ms", 'info')

        # CRITICAL: Initialize camera BEFORE starting cycle
        if self.camera is None:
            self.addLogEntry("Initializing camera system...", 'info')
            try:
                self.init_camera_optimized()
                self.addLogEntry("✅ Camera initialized successfully", 'info')
            except Exception as e:
                self.addLogEntry(f"❌ Camera initialization failed: {e}", 'error')
                self.running = False
                raise Exception(f"Camera initialization failed: {e}")

        # Enable detection
        self.detection_enabled = True
        self.addLogEntry("✅ ML detection enabled", 'info')

        # Connect AWS IoT MQTT
        if iot_publisher:
            if iot_publisher.connect():
                self.addLogEntry("✅ AWS IoT MQTT connected", 'info')
            else:
                self.addLogEntry("⚠️ AWS IoT MQTT connection failed - messages will be saved for retry", 'warning')

        # Start background processing workers
        self.start_background_workers()

        # Start the automated cycle in a new thread
        self.cycle_thread = threading.Thread(target=self.run_automated_cycle, daemon=True)
        self.cycle_thread.start()
        self.addLogEntry("🚀 Automated cycle thread started", 'info')

        return self.run_id

    def async_dispense_trigger(self):
        """Async dispenser trigger - runs in parallel with image capture
        
        Dispenses 6 cards (2 rows × 3 cards each) while camera captures current frame.
        This ensures the NEXT batch is ready when current processing completes.
        """
        with self.dispense_lock:
            if self.dispense_in_progress:
                return  # Already dispensing, skip
            self.dispense_in_progress = True
        
        try:
            self.addLogEntry("🎯 Dispensing next batch (6 cards) in parallel with capture...", 'info')
            dispense_start = time.time()
            self.dispense_sim_cards()
            dispense_time = time.time() - dispense_start
            self.addLogEntry(f"✅ Dispense complete ({dispense_time:.1f}s) - next batch ready", 'info')
            # Note: frames_since_dispense not reset - we dispense every frame
        except Exception as e:
            self.addLogEntry(f"❌ Async dispense failed: {e}", 'error')
        finally:
            with self.dispense_lock:
                self.dispense_in_progress = False

    def async_optimized_capture_loop(self):
        """Async main processing loop - processes frames and uploads individual SIM card crops

        CORRECTED FLOW (Simultaneous Capture + Dispense):
        - Start: Belt has 6 cards ready
        - Frame N: Capture current 6 cards + dispense NEXT 6 cards (parallel)
        - Wait for both capture AND dispense to complete
        - Move belt 10cm (positions next 6 cards under camera)
        - Frame N+1: Repeat with fresh cards

        This ensures belt ALWAYS has 6 cards ready - no gaps!
        """
        self.addLogEntry("Starting SIM card detection and multi-crop upload loop", 'info')
        loop_start = time.time()
        loop_frame_count = 0
        
        # Performance tracking
        cycle_times = []
        last_performance_log = time.time()

        while self.running and not self.should_stop:
            if not self.paused:
                try:
                    loop_frame_count += 1
                    cycle_start = time.time()
                    
                    self.addLogEntry(f"Processing frame {loop_frame_count}", 'info')
                    
                    # Update frame counter
                    with self.image_count_lock:
                        self.frame_count = loop_frame_count
                    
                    # CRITICAL: Start dispense FIRST (parallel with capture)
                    # This ensures next batch is ready while we process current batch
                    dispense_thread = threading.Thread(target=self.async_dispense_trigger, daemon=True)
                    dispense_thread.start()
                    self.addLogEntry("🚀 Started parallel dispense (next 6 cards while capturing)", 'info')
                    
                    # Non-blocking capture and process (uploads multiple crops per frame)
                    success = self.capture_and_process_optimized(loop_frame_count)
                    
                    # Wait for dispense to complete before moving belt
                    # This ensures the belt doesn't move while dispenser is active
                    dispense_thread.join(timeout=5.0)  # 5 second timeout
                    if dispense_thread.is_alive():
                        self.addLogEntry("⚠️ Dispense thread still running - continuing anyway", 'warning')
                    
                    if success:
                        # Allow ML detections to stabilize before moving the conveyor
                        time.sleep(DETECTION_SETTLE_DELAY)
                        self.emit_performance_update()
                        
                        # Success = SIM cards detected and uploaded
                        # NOW move belt forward (safe since dispense completed)
                        self.move_conveyor_belt_optimized(1.0)  # Always 10cm for consistency
                        
                        # Fixed processing delay for stability - belt stabilization is handled in move function
                        time.sleep(PROCESSING_DELAY)
                        
                        cycle_time = time.time() - cycle_start
                        cycle_times.append(cycle_time)
                        
                        # Keep only recent measurements
                        if len(cycle_times) > 50:
                            cycle_times = cycle_times[-50:]
                        
                        # Performance logging every 5 successful captures
                        if len(cycle_times) % 5 == 0:
                            avg_cycle = sum(cycle_times[-5:]) / min(len(cycle_times), 5)
                            frames_per_minute = 60.0 / avg_cycle if avg_cycle > 0 else 0
                            
                            with self.stats_lock:
                                queue_status = f"Upload:{self.upload_queue.qsize()}, Webhook:{self.webhook_queue.qsize()}"
                                error_status = f"Drops:{self.dropped_frames}, QueueFull:{self.queue_full_events}"
                                simcards_total = self.image_count
                                frames_with_detections = len(cycle_times)
                                avg_simcards = simcards_total / frames_with_detections if frames_with_detections > 0 else 0
                            
                            self.addLogEntry(f"Throughput: {frames_per_minute:.1f} frames/min, {simcards_total} SIM cards ({avg_simcards:.1f}/frame with detections), {queue_status}, {error_status}", 'info')
                        
                        # Performance warning if falling behind target
                        if len(cycle_times) > 10:  # After warmup
                            recent_avg = sum(cycle_times[-10:]) / min(len(cycle_times), 10)
                            target_cycle_time = 60.0 / 25  # 25 frames per minute target
                            if recent_avg > target_cycle_time * 1.2:
                                self.addLogEntry(f"Performance warning: {60/recent_avg:.1f} frames/min (target: 25+)", 'warning')
                        
                    else:
                        # No SIM cards detected - keep looking at same spot (belt doesn't move)
                        self.emit_performance_update()
                        time.sleep(0.5)  # Brief pause before checking again

                except Exception as e:
                    self.log_exception(f"Async capture loop iteration {loop_frame_count}", e, 'error')
                    time.sleep(0.5)  # Error recovery delay
            else:
                time.sleep(0.1)  # Short pause when paused

        total_runtime = time.time() - loop_start
        avg_cycle_time = sum(cycle_times) / len(cycle_times) if cycle_times else 0
        actual_throughput = (loop_frame_count / total_runtime) * 60 if total_runtime > 0 else 0
        
        # Final performance summary
        with self.stats_lock:
            final_stats = {
                'frames_processed': loop_frame_count,
                'simcards_uploaded': self.image_count,
                'runtime': total_runtime,
                'avg_cycle_time': avg_cycle_time,
                'actual_throughput': actual_throughput,
                'avg_simcards_per_frame': self.image_count / loop_frame_count if loop_frame_count > 0 else 0,
                'dropped_frames': self.dropped_frames,
                'queue_overflows': self.queue_full_events,
                'upload_errors': self.upload_errors,
                'iot_errors': self.iot_errors,
                'upload_queue_final': self.upload_queue.qsize()
            }
        
        self.addLogEntry("Async capture loop completed", 'info')
        self.addLogEntry(f"Final: {loop_frame_count} frames, {self.image_count} SIM cards, {actual_throughput:.1f} frames/min, {final_stats['dropped_frames']} drops", 'info')
        
        return final_stats

    def stop_system(self):
        """Stop async system with comprehensive performance summary"""
        if not self.running:
            return False

        self.addLogEntry("Initiating system shutdown...", 'info')
        
        self.should_stop = True
        self.running = False
        self.paused = False

        # Stop capture thread
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=5)
            if self.capture_thread.is_alive():
                self.addLogEntry("Capture thread shutdown timeout", 'warning')

        # Use new graceful shutdown method
        try:
            self.stop_background_workers()
        except Exception as e:
            self.log_exception("Worker thread cleanup", e, 'warning')

        # Generate comprehensive performance summary
        if hasattr(self, 'start_time'):
            runtime = datetime.now() - self.start_time
            minutes = runtime.total_seconds() / 60
            
            with self.stats_lock:
                final_image_count = self.image_count
                
                performance_summary = {
                    'total_images': final_image_count,
                    'runtime_minutes': minutes,
                    'images_per_minute': final_image_count / minutes if minutes > 0 else 0,
                    'dropped_frames': self.dropped_frames,
                    'queue_overflows': self.queue_full_events,
                    'upload_errors': self.upload_errors,
                    'iot_errors': self.iot_errors,
                    'avg_capture_time': sum(self.capture_times) / len(self.capture_times) if self.capture_times else 0,
                    'avg_upload_time': sum(self.upload_times) / len(self.upload_times) if self.upload_times else 0
                }
            
            # Log comprehensive summary
            self.addLogEntry("=== SYSTEM PERFORMANCE SUMMARY ===", 'info')
            self.addLogEntry(f"Images Processed: {performance_summary['total_images']}", 'info')
            self.addLogEntry(f"Runtime: {performance_summary['runtime_minutes']:.1f} minutes", 'info')
            self.addLogEntry(f"Throughput: {performance_summary['images_per_minute']:.1f} images/minute", 'info')
            self.addLogEntry(f"Target Achievement: {(performance_summary['images_per_minute']/25)*100:.1f}% of 25 img/min target", 'info')
            self.addLogEntry(f"Reliability: {performance_summary['dropped_frames']} drops, {performance_summary['upload_errors']} upload errors", 'info')
            self.addLogEntry(f"Performance: {performance_summary['avg_capture_time']:.3f}s capture, {performance_summary['avg_upload_time']:.3f}s upload", 'info')
            
            # Send final summary to frontend
            if self.status_callback:
                try:
                    self.status_callback({
                        'type': 'system_stopped',
                        'performance_summary': performance_summary,
                        'timestamp': datetime.now().isoformat()
                    })
                except Exception as callback_error:
                    print(f"Final callback error: {callback_error}")

        # CRITICAL: Cleanup camera resources to prevent future conflicts
        self._cleanup_camera_resources()
        self.addLogEntry("Camera resources cleaned up", 'info')

        return True
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=5)

        return True

    def pause_system(self):
        """Pause system"""
        if not self.running:
            return False
        self.paused = True
        self.addLogEntry("System paused")
        return True

    def resume_system(self):
        """Resume system"""
        if not self.running:
            return False
        self.paused = False
        self.addLogEntry("System resumed")
        return True

    def get_status(self):
        """Get enhanced system status with performance metrics and detection info"""
        # Calculate dispense speed
        dispense_speed = sum(self.dispense_times) / len(self.dispense_times) if self.dispense_times else 0

        # Get latest detection confidence
        detection_confidence = 0
        if hasattr(self, 'detection_stats') and self.detection_stats:
            if 'avg_confidence' in self.detection_stats:
                detection_confidence = self.detection_stats['avg_confidence']
            elif 'last_confidence' in self.detection_stats:
                detection_confidence = self.detection_stats['last_confidence']

        status = {
            'running': self.running,
            'paused': self.paused,
            'image_count': self.image_count,  # Total SIM cards uploaded
            'frame_count': self.frame_count if hasattr(self, 'frame_count') else 0,
            'service_provider': self.service_provider,
            'run_id': self.run_id,
            'last_image_time': datetime.now().isoformat() if self.image_count > 0 else None,
            'start_time': self.start_time.isoformat() if hasattr(self, 'start_time') and self.start_time else None,
            'last_image_path': self.last_image_path if hasattr(self, 'last_image_path') else None,
            'detection_enabled': self.detection_enabled,
            'detection_count': self.detection_count if self.detection_enabled else 0,
            'detection_stats': self.detection_stats if self.detection_enabled else None,
            # Frontend expects these fields at root level
            'sim_cards_detected': self.detection_count if self.detection_enabled else 0,
            'detection_confidence': detection_confidence,
            'dispensed_sim_cards_count': self.dispensed_sim_cards_count,
            'dispense_speed': dispense_speed,
            'stream_stats': {'fps': getattr(self, 'current_fps', 0)}
        }

        # Add performance metrics if available
        if self.capture_times:
            status['performance'] = {
                'avg_capture_time': sum(self.capture_times[-5:]) / min(len(self.capture_times), 5),
                'avg_upload_time': sum(self.upload_times[-5:]) / min(len(self.upload_times), 5) if self.upload_times else 0,
                'total_processing_time': self.total_processing_time,
                'images_processed': len(self.capture_times),
                'dispense_speed': dispense_speed
            }

        # Add system health metrics
        status['health'] = {
            'upload_queue_size': self.upload_queue.qsize() if hasattr(self, 'upload_queue') else 0,
            'iot_queue_size': self.iot_queue.qsize() if hasattr(self, 'iot_queue') else 0,
            'upload_errors': self.upload_errors,
            'iot_errors': self.iot_errors,
            'dropped_frames': self.dropped_frames,
            'queue_full_events': self.queue_full_events,
            'circuit_breaker_state': self.iot_circuit_breaker.get('state', 'closed'),
            'circuit_breaker_failures': self.iot_circuit_breaker.get('failures', 0),
            'failed_iot_messages_pending': len(glob.glob(f"{FAILED_IOT_MESSAGES_DIR}/*.json")),
            'failed_uploads_pending': len(glob.glob(f"{FAILED_UPLOADS_DIR}/*.json")),
            'iot_connected': iot_publisher.connected if iot_publisher else False
        }

        return status

    def emit_performance_update(self):
        """Emit real-time performance metrics to frontend via Socket.IO"""
        if not self.status_callback:
            return

        try:
            with self.stats_lock:
                # Calculate images per minute from recent captures
                recent_window = 60  # 60 second window
                current_time = time.time()
                recent_captures = [ts for ts in self.capture_timestamps if current_time - ts < recent_window]
                images_per_minute = len(recent_captures)

                # Calculate average times
                avg_capture = sum(self.capture_times[-10:]) / min(len(self.capture_times), 10) if self.capture_times else 0
                avg_upload = sum(self.upload_times[-10:]) / min(len(self.upload_times), 10) if self.upload_times else 0

                performance_data = {
                    'avg_capture_time': avg_capture,
                    'avg_upload_time': avg_upload,
                    'images_per_minute': images_per_minute,
                    'total_images': self.image_count,
                    'frame_count': self.frame_count if hasattr(self, 'frame_count') else 0,
                    'dropped_frames': self.dropped_frames,
                    'queue_full_events': self.queue_full_events,
                    'upload_errors': self.upload_errors,
                    'iot_errors': self.iot_errors,
                    'upload_queue_size': self.upload_queue.qsize() if hasattr(self, 'upload_queue') else 0,
                    'iot_queue_size': self.iot_queue.qsize() if hasattr(self, 'iot_queue') else 0
                }

                # Add health metrics
                health_data = {
                    'circuit_breaker_state': self.iot_circuit_breaker.get('state', 'closed'),
                    'circuit_breaker_failures': self.iot_circuit_breaker.get('failures', 0),
                    'failed_iot_messages_pending': len(glob.glob(f"{FAILED_IOT_MESSAGES_DIR}/*.json")),
                    'failed_uploads_pending': len(glob.glob(f"{FAILED_UPLOADS_DIR}/*.json")),
                    'iot_connected': iot_publisher.connected if iot_publisher else False
                }

            # Send performance update via status callback
            self.status_callback({
                'type': 'performance_update',
                'performance': performance_data,
                'health': health_data,
                'timestamp': datetime.now().isoformat()
            })

        except Exception as e:
            print(f"Error emitting performance update: {e}")

    def cleanup(self):
        """Enhanced cleanup with performance summary"""
        try:
            print("Starting optimized system cleanup...")
            
            # Cleanup detection processor
            if hasattr(self, 'detection_processor'):
                try:
                    self.detection_processor.cleanup()
                except Exception as e:
                    print(f"Detection cleanup warning: {e}")
            
            # Print final performance summary
            if self.capture_times:
                total_captures = len(self.capture_times)
                total_capture_time = sum(self.capture_times)
                print(f"Performance Summary: {total_captures} frames, {total_capture_time:.1f}s total capture time")
            
            # Print detection statistics
            if self.detection_enabled and hasattr(self, 'detection_stats'):
                print(f"Detection Stats: {self.detection_stats['total_detections']} total, "
                      f"{self.detection_stats['frames_with_detections']} with detections, "
                      f"{self.detection_stats['frames_without_detections']} without, "
                      f"{self.detection_stats['total_simcards_uploaded']} SIM cards uploaded")
            
            # Stop streaming
            self.stream_active = False
            if hasattr(self, 'stream_thread') and self.stream_thread:
                self.stream_thread.join(timeout=3)
            
            # Stop system
            if self.running:
                self.stop_system()
            
            # Disconnect AWS IoT MQTT
            if iot_publisher:
                try:
                    iot_publisher.disconnect()
                except Exception as e:
                    print(f"⚠️ IoT MQTT disconnect warning: {e}")

            # Cleanup ESP32 connection
            if hasattr(self, 'esp32') and self.esp32:
                try:
                    # Reset servos to home position
                    self.send_esp32_command("RESET", wait_for_ok=False)
                    time.sleep(0.5)
                    self.esp32.close()
                    print("✅ ESP32 connection closed")
                except Exception as e:
                    print(f"⚠️ ESP32 cleanup warning: {e}")
                finally:
                    self.esp32 = None
            
            # Cleanup camera
            if hasattr(self, 'camera') and self.camera:
                try:
                    self.camera.stop()
                    time.sleep(1)
                    self.camera.close()
                    time.sleep(1)
                except Exception as e:
                    print(f"Camera cleanup warning: {e}")
                finally:
                    self.camera = None
            
            # Reset state
            self.stream_buffer = None
            self.stream_active = False
            self.running = False
            self.paused = False
            self.should_stop = True
            
            print("Optimized system cleanup complete")
            return True
            
        except Exception as e:
            print(f"Error during cleanup: {e}")
            return False

def cleanup_camera_system():
    """Cleanup camera system components"""
    cleanup_all_gpio()
    cv2.destroyAllWindows()
    print("Camera system components cleaned up")

# Global system instance with thread safety
camera_system = None
_system_lock = threading.Lock()

def get_system_instance(status_callback=None):
    """Thread-safe singleton system instance"""
    global camera_system
    
    # Thread-safe singleton pattern
    with _system_lock:
        if camera_system is None:
            print("Creating new camera system instance...")
            camera_system = OptimizedCameraConveyorSystem(status_callback)
        else:
            print("Returning existing camera system instance")
    
    return camera_system

def cleanup_system():
    """Thread-safe cleanup of global system instance"""
    global camera_system
    
    with _system_lock:
        try:
            if camera_system:
                print("Cleaning up camera system instance...")
                camera_system.cleanup()
                camera_system = None
            cleanup_camera_system()
        except Exception as e:
            print(f"Error during system cleanup: {e}")

def emergency_cleanup():
    """Emergency cleanup for signal handlers"""
    try:
        cleanup_system()
    except:
        pass

# Register cleanup handlers
atexit.register(emergency_cleanup)
signal.signal(signal.SIGTERM, lambda s, f: emergency_cleanup())
signal.signal(signal.SIGINT, lambda s, f: emergency_cleanup())

print("Optimized camera system module loaded - Ready for OCR processing")
