#!/usr/bin/env python3
"""
SIM Card Detection System - Clean Detection Pipeline
Single-stage Edge Impulse SSD detection with dual resolution support
Cleaned of OCR extraction attempts - ready for cloud OCR integration
"""

import pygame
import cv2
import numpy as np
import time
import os
import sys
import signal
from datetime import datetime
import threading
import queue
import math

# Edge Impulse imports
try:
    from edge_impulse_linux.image import ImageImpulseRunner
    EDGE_IMPULSE_AVAILABLE = True
    print("✅ Edge Impulse Linux SDK loaded successfully")
except ImportError as e:
    EDGE_IMPULSE_AVAILABLE = False
    print(f"❌ Edge Impulse Linux SDK not available: {e}")
    print("📦 Install with: pip install edge-impulse-linux")
    sys.exit(1)

# Camera imports
try:
    from picamera2 import Picamera2
    RASPBERRY_PI_MODE = True
    print("🍓 Raspberry Pi camera mode enabled")
except ImportError:
    RASPBERRY_PI_MODE = False
    print("💻 Development mode - Using OpenCV camera")

# Initialize Pygame
pygame.init()

# UI Configuration
WINDOW_WIDTH = 1600
WINDOW_HEIGHT = 1000
# Dual resolution support: High-quality visuals while maintaining ML compatibility
DISPLAY_WIDTH = 1024     # High resolution for excellent visual assessment
DISPLAY_HEIGHT = 1024    # Square format for consistency  
ML_WIDTH = 320           # ML model requirement - detection stage
ML_HEIGHT = 320          # ML model requirement - detection stage
BUTTON_HEIGHT = 50
BUTTON_WIDTH = 200
MARGIN = 20

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
LIGHT_GRAY = (200, 200, 200)
DARK_GRAY = (64, 64, 64)
GREEN = (0, 200, 0)
RED = (200, 0, 0)
BLUE = (0, 100, 200)
ORANGE = (255, 165, 0)
PURPLE = (128, 0, 128)

# Global variables for shutdown handling
show_camera = True
detection_runner = None

def signal_handler(sig, frame):
    """Clean shutdown on SIGINT"""
    global show_camera, detection_runner
    print('🛑 Interrupted - Shutting down...')
    show_camera = False
    if detection_runner:
        detection_runner.stop()
    pygame.quit()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def draw_detection_results(frame, detections):
    """Draw bounding boxes and labels for detected SIM cards"""
    for detection in detections:
        bbox = detection['bbox']
        x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
        
        # Detection info
        sim_type = detection.get('label', 'Unknown')
        confidence = detection.get('confidence', 0.0)
        
        # Choose color based on detection type for FOMO
        if 'simcard' in sim_type.lower():
            color = (0, 255, 0)    # Green for SIM cards
        elif 'background' in sim_type.lower():
            color = (128, 128, 128)  # Gray for background
        else:
            color = (255, 0, 0)    # Blue for unknown
        
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
        
        # Create detection label
        label = f"{sim_type} ({confidence:.2f})"
        
        # Calculate text size and position
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Position label above bounding box
        label_y = y - 15 if y > 30 else y + h + 30
        
        # Draw label background
        cv2.rectangle(frame, (x, label_y - text_h - 5), (x + text_w + 10, label_y + 5), color, -1)
        
        # Draw label
        cv2.putText(frame, label, (x + 5, label_y - 5), font, font_scale, (255, 255, 255), thickness)

class DetectionProcessor:
    """SIM Card Detection Processor using Edge Impulse SSD model"""
    
    def __init__(self):
        self.detection_runner = None
        self.detection_labels = []
        self.detection_input_width = 0
        self.detection_input_height = 0
        self.initialized = False
        
        # Detection filtering parameters
        self.confidence_threshold = 0.3  # Lower threshold to see more detections (was 0.6)
        self.nms_threshold = 0.4  # IoU threshold for NMS
        self.debug_mode = True  # Enable debug mode by default to see all detections
        self.force_rgb_mode = True
        
    def initialize_models(self):
        """Initialize Edge Impulse detection model"""
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            detection_model_path = os.path.join(script_dir, "models", "simcard_detection.eim")
            
            print(f"🔍 Looking for model at: {detection_model_path}")
            print(f"🔍 Model exists: {os.path.exists(detection_model_path)}")
            if os.path.exists(detection_model_path):
                print(f"🔍 Model size: {os.path.getsize(detection_model_path)} bytes")
            
            if not os.path.exists(detection_model_path):
                raise FileNotFoundError(f"SIM detection model not found: {detection_model_path}")
            
            print(f"📁 Loading detection model: {os.path.basename(detection_model_path)}")
            print(f"🔍 Creating ImageImpulseRunner...")
            self.detection_runner = ImageImpulseRunner(detection_model_path)
            print(f"🔍 Initializing model...")
            detection_model_info = self.detection_runner.init()
            
            print(f'✅ Detection model loaded: "{detection_model_info["project"]["owner"]} / {detection_model_info["project"]["name"]}"')
            self.detection_labels = detection_model_info['model_parameters']['labels']
            self.detection_input_width = detection_model_info['model_parameters']['image_input_width']
            self.detection_input_height = detection_model_info['model_parameters']['image_input_height']
            print(f"🏷️ Detection labels: {self.detection_labels}")
            print(f"📐 Detection input size: {self.detection_input_width}x{self.detection_input_height}")
            print(f"🔍 Current confidence threshold: {self.confidence_threshold}")
            
            self.initialized = True
            return True
            
        except Exception as e:
            print(f"❌ Model initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
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
    
    def apply_nms(self, detections):
        """Apply Non-Maximum Suppression to filter duplicate detections"""
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
            
            # Check if this detection overlaps significantly with any already accepted detection
            should_keep = True
            for accepted in filtered_detections:
                iou = self.calculate_iou(current, accepted)
                if iou > self.nms_threshold:
                    # If same class, keep the higher confidence one (already sorted)
                    if current['label'] == accepted['label']:
                        print(f"🔄 NMS: Duplicate {current['label']} removed (IoU: {iou:.2f})")
                        should_keep = False
                        break
                    # If different classes but high overlap, keep both but note the conflict
                    else:
                        print(f"⚠️ Class conflict: {current['label']} vs {accepted['label']} (IoU: {iou:.2f})")
            
            if should_keep:
                filtered_detections.append(current)
                print(f"✅ Kept detection: {current['label']} ({current['value']:.2f})")
        
        print(f"📊 NMS Results: {len(detections)} → {len(filtered_detections)} detections")
        return filtered_detections
    
    def process_frame(self, display_frame):
        """
        Single-stage processing with dual resolution:
        1. Downscale to 320x320 for SIM card detection
        2. Scale detection coordinates back to display resolution
        """
        if not self.initialized:
            return []
        
        results = []
        
        try:
            # Prepare ML-sized frame for detection
            ml_frame = cv2.resize(display_frame, (ML_WIDTH, ML_HEIGHT), interpolation=cv2.INTER_AREA)
            
            # SIM Card Detection on ML resolution
            print(f"🔍 Running SIM card detection on {ML_WIDTH}x{ML_HEIGHT} frame...")
            
            # Debug: Check frame formats
            if self.debug_mode:
                print(f"🔍 Display frame shape: {display_frame.shape}")
                print(f"🔍 ML frame shape: {ml_frame.shape}")
                print(f"🔍 Frame dtype: {ml_frame.dtype}")
                print(f"🔍 Frame range: min={ml_frame.min()}, max={ml_frame.max()}")
                print(f"🔍 RASPBERRY_PI_MODE: {RASPBERRY_PI_MODE}")
                print(f"🔍 Force RGB mode: {getattr(self, 'force_rgb_mode', False)}")
            
            # Handle color space for Edge Impulse (models expect RGB input)
            if hasattr(self, 'force_rgb_mode') and self.force_rgb_mode and RASPBERRY_PI_MODE:
                # PiCamera2 in RGB mode - use ML frame directly
                rgb_frame = ml_frame
                if self.debug_mode:
                    print(f"🔍 Using RGB ML frame directly (force_rgb_mode)")
            elif ml_frame.shape[2] == 3:
                # Convert BGR to RGB for Edge Impulse
                rgb_frame = cv2.cvtColor(ml_frame, cv2.COLOR_BGR2RGB)
                if self.debug_mode:
                    print(f"🔍 Converted BGR->RGB for Edge Impulse")
            else:
                rgb_frame = ml_frame
                if self.debug_mode:
                    print(f"🔍 Using ML frame as-is (not 3 channels)")
            
            if self.debug_mode:
                print(f"🔍 RGB frame shape: {rgb_frame.shape}")
                print(f"🔍 RGB frame range: min={rgb_frame.min()}, max={rgb_frame.max()}")
            
            detection_features, processed_img = self.detection_runner.get_features_from_image(rgb_frame)
            detection_result = self.detection_runner.classify(detection_features)
            
            processing_time = detection_result['timing']['dsp'] + detection_result['timing']['classification']
            print(f'📊 Detection: Found {len(detection_result.get("result", {}).get("bounding_boxes", []))} bounding boxes ({processing_time} ms.)')
            
            # Get raw detections
            raw_detections = detection_result.get("result", {}).get("bounding_boxes", [])
            
            # Debug: Show all raw detections before filtering
            if self.debug_mode:
                print(f"🔍 DEBUG: Raw detections count: {len(raw_detections)}")
                for i, bbox in enumerate(raw_detections):
                    print(f"   Detection {i+1}: {bbox['label']} confidence={bbox['value']:.3f} x={bbox['x']} y={bbox['y']} w={bbox['width']} h={bbox['height']}")
            
            # Apply NMS filtering
            filtered_detections = self.apply_nms(raw_detections)
            
            # Debug: Show filtered results
            if self.debug_mode and filtered_detections:
                print(f"🔍 DEBUG: After NMS filtering: {len(filtered_detections)} detections remain")
                for det in filtered_detections:
                    print(f"   Final: {det['label']} confidence={det['value']:.3f}")
            elif self.debug_mode:
                print(f"🔍 DEBUG: No detections after filtering (confidence threshold: {self.confidence_threshold})")
            
            # Process each filtered SIM card detection with coordinate scaling
            for bbox in filtered_detections:
                # Scale coordinates from ML resolution to display resolution
                scaled_bbox = self.scale_coordinates_to_display(bbox, ml_frame.shape, display_frame.shape)
                
                sim_detection = {
                    'label': bbox['label'],
                    'confidence': bbox['value'],
                    'bbox': scaled_bbox,  # Use scaled coordinates for display
                    'ocr_result': 'SIM Card Detected'  # Simple detection confirmation
                }
                
                print(f'📋 Detection: {bbox["label"]} (confidence: {bbox["value"]:.2f})')
                print(f'📐 Coordinates: ML({bbox["x"]},{bbox["y"]},{bbox["width"]},{bbox["height"]}) → Display({scaled_bbox["x"]},{scaled_bbox["y"]},{scaled_bbox["width"]},{scaled_bbox["height"]})')
                
                results.append(sim_detection)
                    
        except Exception as e:
            print(f"❌ Processing error: {e}")
            
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
            self.detection_runner.stop()

class CameraController:
    """Camera controller supporting both Pi and OpenCV modes"""
    
    def __init__(self):
        self.camera = None
        self.running = False
        self.frame = None
        self.frame_lock = threading.Lock()
        
    def init_camera(self):
        """Initialize camera based on platform"""
        try:
            if RASPBERRY_PI_MODE:
                return self.init_picamera2()
            else:
                return self.init_opencv_camera()
        except Exception as e:
            print(f"❌ Camera initialization failed: {e}")
            return False
    
    def init_picamera2(self):
        """Initialize Raspberry Pi camera with dual resolution support"""
        try:
            self.camera = Picamera2()
            
            # Configure for DISPLAY resolution (1024x1024) for excellent visual quality
            config = self.camera.create_preview_configuration(
                main={"size": (DISPLAY_WIDTH, DISPLAY_HEIGHT), "format": "RGB888"}
            )
            self.camera.configure(config)
            
            # OV5647 IR Night Vision Camera Settings
            print("📹 Configuring OV5647 IR camera for dual resolution...")
            print(f"📺 Display resolution: {DISPLAY_WIDTH}x{DISPLAY_HEIGHT}")
            print(f"🤖 ML processing resolution: {ML_WIDTH}x{ML_HEIGHT}")
            
            # Start camera first
            self.camera.start()
            time.sleep(1)  # Initial startup
            
            # Set controls for IR optimization
            controls = {
                # Exposure settings for IR lighting
                "ExposureTime": 20000,  # 20ms exposure for dim IR lighting (F3 preset)
                "AnalogueGain": 6.0,    # Higher gain for dim IR sensitivity
                
                # Contrast and brightness for IR imaging
                "Contrast": 2.2,        # Higher contrast for optimal IR text detection (F3 preset)
                "Brightness": 0.3,      # Higher brightness for dim conditions
                "Saturation": 0.8,      # Reduced saturation for IR
                
                # Focus and sharpness
                "Sharpness": 2.0,       # Enhanced sharpness for text clarity
                
                # White balance (important for IR)
                "AwbEnable": False,     # Disable auto white balance for IR
                "ColourGains": (1.4, 1.2),  # Manual color gains for IR
                
                # Noise reduction
                "NoiseReductionMode": 2,  # Moderate noise reduction
            }
            
            try:
                self.camera.set_controls(controls)
                print("✅ OV5647 IR camera controls applied successfully")
                
                # Log applied settings
                current_controls = self.camera.capture_metadata()
                print(f"📊 Applied settings:")
                print(f"   Exposure: {controls['ExposureTime']}μs")
                print(f"   Gain: {controls['AnalogueGain']}x")
                print(f"   Contrast: {controls['Contrast']}")
                print(f"   Brightness: {controls['Brightness']}")
                print(f"   Sharpness: {controls['Sharpness']}")
                
            except Exception as ctrl_e:
                print(f"⚠️ Some camera controls may not be supported: {ctrl_e}")
                print("📹 Camera will use default settings")
            
            time.sleep(2)  # Camera stabilization after settings
            print("✅ Picamera2 with OV5647 IR optimization initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Picamera2 initialization failed: {e}")
            return False
    
    def init_opencv_camera(self):
        """Initialize OpenCV camera for development with display resolution"""
        try:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                return False
            
            # Set to DISPLAY resolution for better visual quality
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_WIDTH)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_HEIGHT)
            print(f"✅ OpenCV camera initialized at {DISPLAY_WIDTH}x{DISPLAY_HEIGHT}")
            return True
        except Exception as e:
            print(f"❌ OpenCV camera initialization failed: {e}")
            return False
    
    def adjust_ir_settings(self, exposure_time=None, gain=None, contrast=None, brightness=None):
        """Adjust IR camera settings during runtime"""
        if not RASPBERRY_PI_MODE or not self.camera:
            print("⚠️ IR settings adjustment only available on Raspberry Pi")
            return False
        
        try:
            controls = {}
            
            if exposure_time is not None:
                controls["ExposureTime"] = exposure_time
                print(f"📹 Setting exposure: {exposure_time}μs")
            
            if gain is not None:
                controls["AnalogueGain"] = gain
                print(f"📹 Setting gain: {gain}x")
            
            if contrast is not None:
                controls["Contrast"] = contrast
                print(f"📹 Setting contrast: {contrast}")
            
            if brightness is not None:
                controls["Brightness"] = brightness
                print(f"📹 Setting brightness: {brightness}")
            
            if controls:
                self.camera.set_controls(controls)
                print("✅ IR camera settings updated")
                return True
            
            return False
            
        except Exception as e:
            print(f"❌ Failed to adjust IR settings: {e}")
            return False
    
    def get_ir_presets(self):
        """Get predefined IR camera settings for different conditions"""
        return {
            "bright_ir": {
                "ExposureTime": 5000,   # 5ms for bright IR lights
                "AnalogueGain": 2.0,
                "Contrast": 1.5,
                "Brightness": 0.1
            },
            "normal_ir": {
                "ExposureTime": 10000,  # 10ms for normal IR lights
                "AnalogueGain": 4.0,
                "Contrast": 2.2,        # Updated to match new permanent setting
                "Brightness": 0.2
            },
            "dim_ir": {
                "ExposureTime": 20000,  # 20ms for dim IR lights
                "AnalogueGain": 6.0,
                "Contrast": 2.2,
                "Brightness": 0.3
            }
        }
    
    def apply_ir_preset(self, preset_name):
        """Apply a predefined IR preset"""
        presets = self.get_ir_presets()
        if preset_name not in presets:
            print(f"❌ Unknown preset: {preset_name}")
            print(f"Available presets: {list(presets.keys())}")
            return False
        
        preset = presets[preset_name]
        print(f"📹 Applying IR preset: {preset_name}")
        
        return self.adjust_ir_settings(
            exposure_time=preset["ExposureTime"],
            gain=preset["AnalogueGain"],
            contrast=preset["Contrast"],
            brightness=preset["Brightness"]
        )

    def capture_frame(self):
        """Capture a frame from the camera"""
        try:
            with self.frame_lock:  # Thread-safe capture
                if RASPBERRY_PI_MODE and self.camera:
                    frame = self.camera.capture_array()
                    
                    # RGB mode test: Skip BGR conversion for Edge Impulse compatibility
                    if hasattr(self, 'force_rgb_mode') and self.force_rgb_mode:
                        if hasattr(self, 'debug_mode') and self.debug_mode:
                            print(f"🎥 PiCamera2: Keeping RGB888 {frame.shape} for Edge Impulse")
                        return frame  # Keep RGB for Edge Impulse
                    else:
                        # Standard: Convert to BGR for OpenCV compatibility
                        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        if hasattr(self, 'debug_mode') and self.debug_mode:
                            print(f"🎥 PiCamera2: RGB888 {frame.shape} -> BGR {bgr_frame.shape}")
                        return bgr_frame
                        
                elif self.camera:
                    ret, frame = self.camera.read()
                    if hasattr(self, 'debug_mode') and self.debug_mode and ret:
                        print(f"🎥 OpenCV: BGR {frame.shape}")
                    return frame if ret else None
            return None
        except Exception as e:
            print(f"❌ Frame capture error: {e}")
            return None
    
    def cleanup(self):
        """Clean up camera resources"""
        try:
            if RASPBERRY_PI_MODE and self.camera:
                self.camera.stop()
                self.camera.close()
            elif self.camera:
                self.camera.release()
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"⚠️ Camera cleanup warning: {e}")

class OCRTestUI:
    """PyGame UI for OCR Edge AI testing"""
    
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("OCR Test - SIM Detection + OCR")
        
        self.font = pygame.font.Font(None, 24)
        self.title_font = pygame.font.Font(None, 36)
        
        self.camera = CameraController()
        self.processor = DetectionProcessor()
        
        self.running = False
        self.processing_enabled = False
        self.capture_only_mode = False  # New: Pure capture mode for training data
        self.capture_counter = 0  # Track captured images
        self.last_results = []
        self.stats = {
            'frames_processed': 0,
            'detections_found': 0,
            'processing_time': 0,
            'images_captured': 0
        }
        
        # Organized training data structure for FOMO
        self.setup_training_directories()
        self.classification_counts = {
            'simcard': 0,
            'background': 0,
            'raw': 0
        }
    
    def initialize(self):
        """Initialize all components"""
        print("🚀 Initializing OCR Test UI...")
        
        if not self.camera.init_camera():
            print("❌ Camera initialization failed")
            return False
        
        if not self.capture_only_mode:
            if not self.processor.initialize_models():
                print("❌ Model initialization failed - falling back to capture-only mode")
                self.capture_only_mode = True
        else:
            print("📷 Capture-only mode: Skipping AI model initialization")
        
        print("✅ All components initialized successfully")
        return True
    
    def setup_training_directories(self):
        """Create organized folder structure for FOMO training data"""
        base_dir = "training_data"
        self.training_dirs = {
            'simcard': os.path.join(base_dir, "simcard"),
            'background': os.path.join(base_dir, "background"),
            'raw': os.path.join(base_dir, "raw")
        }
        
        # Create directories
        for class_name, dir_path in self.training_dirs.items():
            os.makedirs(dir_path, exist_ok=True)
            print(f"📁 Created directory: {dir_path}")
    
    def save_classified_image(self, frame, classification):
        """Save image with classification-based naming and folder organization"""
        if classification not in self.training_dirs:
            print(f"❌ Unknown classification: {classification}")
            return None
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        self.classification_counts[classification] += 1
        count = self.classification_counts[classification]
        
        filename = f"{classification}_{timestamp}_{count:03d}.jpg"
        filepath = os.path.join(self.training_dirs[classification], filename)
        
        # Save image
        cv2.imwrite(filepath, frame)
        self.stats['images_captured'] += 1
        
        print(f"📷 Saved {classification}: {filename}")
        print(f"📊 Counts - SIM Card: {self.classification_counts['simcard']}, Background: {self.classification_counts['background']}, Raw: {self.classification_counts['raw']}")
        
        return filepath
    
    def draw_ui(self):
        """Draw the main UI"""
        self.screen.fill(WHITE)
        
        # Title with mode indicator
        mode_text = " - CAPTURE MODE (Save Images)" if self.capture_only_mode else " - AI LIVE MODE (No Saving)"
        title_text = self.title_font.render(f"OCR Test - Edge AI{mode_text}", True, BLACK)
        self.screen.blit(title_text, (MARGIN, MARGIN))
        
        # Status info
        status_y = 60
        status_texts = [
            f"Camera: {'✅ Active' if self.camera.camera else '❌ Failed'}",
            f"Mode: {'📷 Capture Only' if self.capture_only_mode else '🤖 AI Processing'}",
            f"Models: {'⏭️ Skipped' if self.capture_only_mode else ('✅ Loaded' if self.processor.initialized else '❌ Failed')}",
            f"Processing: {'📷 Capture Mode (Saves Images)' if self.capture_only_mode else ('🤖 Live AI Mode (No Saving)' if self.processing_enabled else '⏸️ Paused')}",
            f"Total Images: {self.stats['images_captured']}",
            f"SIM Card: {self.classification_counts['simcard']} | Background: {self.classification_counts['background']} | Raw: {self.classification_counts['raw']}",
            f"Frames Processed: {self.stats['frames_processed']}, Detections: {self.stats['detections_found']}"
        ]
        
        for i, text in enumerate(status_texts):
            color = PURPLE if "Capture" in text else BLACK
            if "MTN:" in text:  # Color code the classification counts
                color = DARK_GRAY
            rendered = self.font.render(text, True, color)
            self.screen.blit(rendered, (MARGIN, status_y + i * 25))
        
        # Control buttons - simplified for FOMO training data collection
        button_start_x = WINDOW_WIDTH - (3 * (BUTTON_WIDTH + 10)) - MARGIN
        button_y = 60  # Moved to top area
        if self.capture_only_mode:
            self.draw_button("📷 Background (B)", button_start_x, button_y, GREEN)
            self.draw_button("� SIM Card (S)", button_start_x + BUTTON_WIDTH + 10, button_y, BLUE)
            self.draw_button("🤖 Live AI", button_start_x + 2 * (BUTTON_WIDTH + 10), button_y, ORANGE)
            
            # Second row of buttons
            button_y2 = button_y + BUTTON_HEIGHT + 10
            self.draw_button("📄 Raw (C)", button_start_x, button_y2, GRAY)
            self.draw_button(" Reset Stats", button_start_x + BUTTON_WIDTH + 10, button_y2, PURPLE)
            self.draw_button("Quit", button_start_x + 2 * (BUTTON_WIDTH + 10), button_y2, DARK_GRAY)
        else:
            self.draw_button("Toggle Processing", button_start_x, button_y, GREEN if self.processing_enabled else RED)
            self.draw_button("Capture & Save", button_start_x + BUTTON_WIDTH + 10, button_y, BLUE)
            self.draw_button("📷 Capture Mode", button_start_x + 2 * (BUTTON_WIDTH + 10), button_y, PURPLE)
            self.draw_button("Quit", button_start_x + 3 * (BUTTON_WIDTH + 10), button_y, DARK_GRAY)
        
        # Results display
        results_y = 280
        
        # Add keyboard shortcuts help for capture mode
        if self.capture_only_mode:
            help_text1 = "Single: S=SIM Card, B=Background, C=Raw, R=Reset, Q=Quit"
            help_text2 = "Batch (10x): 1=Background, 2=SIM Card, 4=Raw"
            help_text3 = "IR Camera: F1=Bright, F2=Normal, F3=Dim, F5/F6=Contrast, F7/F8=Exposure"
            help_text4 = "Debug: F9=Toggle Debug, F10/F11=Lower/Raise Threshold"
            help_rendered1 = self.font.render(help_text1, True, PURPLE)
            help_rendered2 = self.font.render(help_text2, True, PURPLE) 
            help_rendered3 = self.font.render(help_text3, True, BLUE)
            help_rendered4 = self.font.render(help_text4, True, ORANGE)
            self.screen.blit(help_rendered1, (MARGIN, results_y))
            self.screen.blit(help_rendered2, (MARGIN, results_y + 20))
            self.screen.blit(help_rendered3, (MARGIN, results_y + 40))
            self.screen.blit(help_rendered4, (MARGIN, results_y + 60))
            results_y += 85
        
        results_title = self.font.render("Latest Detection Results:", True, BLACK)
        self.screen.blit(results_title, (MARGIN, results_y))
        
        for i, detection in enumerate(self.last_results[:5]):  # Show last 5 results
            result_text = f"{detection.get('label', 'Unknown')}: {detection.get('confidence', 0):.2f}"
            rendered = self.font.render(result_text, True, DARK_GRAY)
            self.screen.blit(rendered, (MARGIN, results_y + 30 + i * 25))
    
    def draw_button(self, text, x, y, color):
        """Draw a button with text"""
        pygame.draw.rect(self.screen, color, (x, y, BUTTON_WIDTH, BUTTON_HEIGHT))
        pygame.draw.rect(self.screen, BLACK, (x, y, BUTTON_WIDTH, BUTTON_HEIGHT), 2)
        
        text_surface = self.font.render(text, True, WHITE if color != WHITE else BLACK)
        text_rect = text_surface.get_rect(center=(x + BUTTON_WIDTH//2, y + BUTTON_HEIGHT//2))
        self.screen.blit(text_surface, text_rect)
        
        return pygame.Rect(x, y, BUTTON_WIDTH, BUTTON_HEIGHT)
    
    def handle_button_click(self, pos):
        """Handle button clicks"""
        button_start_x = WINDOW_WIDTH - (4 * (BUTTON_WIDTH + 10)) - MARGIN
        button_y = 60  # Moved to top area
        button_y2 = button_y + BUTTON_HEIGHT + 10
        
        # First row buttons
        buttons_row1 = [
            pygame.Rect(button_start_x, button_y, BUTTON_WIDTH, BUTTON_HEIGHT),  # Button 1
            pygame.Rect(button_start_x + BUTTON_WIDTH + 10, button_y, BUTTON_WIDTH, BUTTON_HEIGHT),  # Button 2
            pygame.Rect(button_start_x + 2 * (BUTTON_WIDTH + 10), button_y, BUTTON_WIDTH, BUTTON_HEIGHT),  # Button 3
            pygame.Rect(button_start_x + 3 * (BUTTON_WIDTH + 10), button_y, BUTTON_WIDTH, BUTTON_HEIGHT),  # Button 4
        ]
        
        # Second row buttons (capture mode only)
        buttons_row2 = [
            pygame.Rect(button_start_x, button_y2, BUTTON_WIDTH, BUTTON_HEIGHT),  # Raw
            pygame.Rect(button_start_x + BUTTON_WIDTH + 10, button_y2, BUTTON_WIDTH, BUTTON_HEIGHT),  # Batch BG
            pygame.Rect(button_start_x + 2 * (BUTTON_WIDTH + 10), button_y2, BUTTON_WIDTH, BUTTON_HEIGHT),  # Reset
            pygame.Rect(button_start_x + 3 * (BUTTON_WIDTH + 10), button_y2, BUTTON_WIDTH, BUTTON_HEIGHT),  # Quit
        ]
        
        if self.capture_only_mode:
            # First row - capture mode
            if buttons_row1[0].collidepoint(pos):  # Background
                frame = self.camera.capture_frame()
                if frame is not None:
                    self.save_classified_image(frame, 'background')
            elif buttons_row1[1].collidepoint(pos):  # SIM Card
                frame = self.camera.capture_frame()
                if frame is not None:
                    self.save_classified_image(frame, 'simcard')
            elif buttons_row1[2].collidepoint(pos):  # Switch to AI
                self.switch_to_ai_mode()
            
            # Second row - capture mode
            elif buttons_row2[0].collidepoint(pos):  # Raw
                frame = self.camera.capture_frame()
                if frame is not None:
                    self.save_classified_image(frame, 'raw')
            elif buttons_row2[1].collidepoint(pos):  # Batch Background
                self.batch_capture_classified('background', 10)
            elif buttons_row2[2].collidepoint(pos):  # Reset Stats
                self.reset_stats()
            elif buttons_row2[3].collidepoint(pos):  # Quit
                self.running = False
        else:
            # AI processing mode
            if buttons_row1[0].collidepoint(pos):
                self.processing_enabled = not self.processing_enabled
                print(f"🔄 Processing {'enabled' if self.processing_enabled else 'disabled'}")
            elif buttons_row1[1].collidepoint(pos):
                self.capture_and_save()
            elif buttons_row1[2].collidepoint(pos):
                self.switch_to_capture_mode()
            elif buttons_row1[3].collidepoint(pos):
                self.running = False
    
    def capture_training_image(self):
        """Capture a clean image for training data (no AI processing)"""
        frame = self.camera.capture_frame()
        if frame is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
            filename = f"training_data_{timestamp}_{self.capture_counter:04d}.jpg"
            
            # Save raw image without any processing
            cv2.imwrite(filename, frame)
            
            # Update stats
            self.capture_counter += 1
            self.stats['images_captured'] += 1
            
            print(f"📷 Training image saved: {filename}")
            print(f"📊 Total training images captured: {self.stats['images_captured']}")
        else:
            print("❌ Failed to capture frame")
    
    def batch_capture(self, count=10):
        """Capture multiple images quickly for training data"""
        print(f"📸 Starting batch capture of {count} images...")
        
        for i in range(count):
            frame = self.camera.capture_frame()
            if frame is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                filename = f"batch_{timestamp}_{i+1:02d}_of_{count:02d}.jpg"
                
                cv2.imwrite(filename, frame)
                self.capture_counter += 1
                self.stats['images_captured'] += 1
                
                print(f"  📷 {i+1}/{count}: {filename}")
                time.sleep(0.5)  # Brief pause between captures
            else:
                print(f"  ❌ Failed to capture frame {i+1}/{count}")
        
        print(f"⏩ Batch capture completed: {count} images")
        print(f"📊 Total training images: {self.stats['images_captured']}")
    
    def batch_capture_classified(self, classification, count=10):
        """Capture multiple images of specified classification quickly"""
        print(f"📸 Starting batch capture of {count} {classification} images...")
        
        for i in range(count):
            frame = self.camera.capture_frame()
            if frame is not None:
                filepath = self.save_classified_image(frame, classification)
                if filepath:
                    print(f"  📷 {i+1}/{count}: {os.path.basename(filepath)}")
                time.sleep(0.5)  # Brief pause between captures
            else:
                print(f"  ❌ Failed to capture frame {i+1}/{count}")
        
        print(f"✅ Batch {classification} capture completed: {count} images")
        print(f"📊 {classification} total: {self.classification_counts[classification]}")
    
    def switch_to_capture_mode(self):
        """Switch to pure capture mode for training data collection"""
        self.capture_only_mode = True
        self.processing_enabled = False
        print("📷 Switched to CAPTURE MODE - AI processing disabled")
        print("🎯 Perfect for collecting training data!")
    
    def switch_to_ai_mode(self):
        """Switch back to AI processing mode"""
        self.capture_only_mode = False
        if not self.processor.initialized:
            print("🤖 Initializing AI models for processing mode...")
            if not self.processor.initialize_models():
                print("❌ Failed to initialize AI models")
                return
        print("🤖 Switched to AI PROCESSING MODE")
        print("🔍 Ready for SIM detection and OCR!")
    
    def capture_and_save(self):
        """Capture frame and save with processing results"""
        frame = self.camera.capture_frame()
        if frame is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Process frame
            if self.processing_enabled:
                results = self.processor.process_frame(frame)
                draw_detection_results(frame, results)
                filename = f"detection_{timestamp}.jpg"
            else:
                filename = f"raw_capture_{timestamp}.jpg"
            
            # Save image
            cv2.imwrite(filename, frame)
            print(f"💾 Saved: {filename}")
    
    def reset_stats(self):
        """Reset processing statistics and classification counts"""
        self.stats = {
            'frames_processed': 0, 
            'detections_found': 0, 
            'processing_time': 0,
            'images_captured': 0
        }
        self.capture_counter = 0
        self.classification_counts = {
            'MTN': 0,
            'Vodacom': 0,
            'background': 0,
            'raw': 0
        }
        self.last_results = []
        print("🔄 Statistics and classification counts reset")
    
    def run(self):
        """Main application loop"""
        if not self.initialize():
            return
        
        self.running = True
        clock = pygame.time.Clock()
        
        print("🎮 OCR Test UI started with dual resolution support")
        print(f"📺 Display resolution: {DISPLAY_WIDTH}x{DISPLAY_HEIGHT} for clear visuals")
        print(f"🤖 ML processing: {ML_WIDTH}x{ML_HEIGHT} for model compatibility")
        print("\n📸 CAPTURE TRAINING DATA:")
        print("  S = Save to 'simcard' folder (multiple SIM cards in frame)")
        print("  B = Save to 'background' folder (empty/no SIM cards)")
        print("  C = Save to 'raw' folder (uncategorized)")
        print("  2 = Batch capture 10 to 'simcard' folder")
        print("  1 = Batch capture 10 to 'background' folder")
        print("\n🎮 CONTROLS:")
        print("  SPACE = Toggle AI processing ON/OFF")
        print("  ESC/Q = Quit")
        print("  F9 = Toggle debug mode")
        print("  F10/F11 = Decrease/Increase confidence threshold")
        
        while self.running and show_camera:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.processing_enabled = not self.processing_enabled
                    elif event.key == pygame.K_ESCAPE:
                        self.running = False
                    # Capture shortcuts - ALWAYS ACTIVE
                    elif event.key == pygame.K_s:  # SIM Card
                        frame = self.camera.capture_frame()
                        if frame is not None:
                            self.save_classified_image(frame, 'simcard')
                    elif event.key == pygame.K_b:  # Background
                        frame = self.camera.capture_frame()
                        if frame is not None:
                            self.save_classified_image(frame, 'background')
                    elif event.key == pygame.K_c:  # Raw (C for capture)
                        frame = self.camera.capture_frame()
                        if frame is not None:
                            self.save_classified_image(frame, 'raw')
                    # Batch capture shortcuts - ALWAYS ACTIVE
                    elif event.key == pygame.K_1:  # Batch background (10 images)
                        self.batch_capture_classified('background', 10)
                    elif event.key == pygame.K_2:  # Batch SIM Card (10 images)  
                        self.batch_capture_classified('simcard', 10)
                    elif event.key == pygame.K_4:  # Batch Raw (10 images)
                        self.batch_capture_classified('raw', 10)
                    # Additional shortcuts
                    elif event.key == pygame.K_r:  # Reset stats
                        self.reset_stats()
                    elif event.key == pygame.K_q:  # Quit
                        self.running = False
                    # IR Camera adjustment shortcuts (F-keys)
                    elif event.key == pygame.K_F1:  # Bright IR preset
                        self.camera.apply_ir_preset('bright_ir')
                    elif event.key == pygame.K_F2:  # Normal IR preset
                        self.camera.apply_ir_preset('normal_ir') 
                    elif event.key == pygame.K_F3:  # Dim IR preset
                        self.camera.apply_ir_preset('dim_ir')
                    elif event.key == pygame.K_F5:  # Increase contrast
                        self.camera.adjust_ir_settings(contrast=2.2)
                    elif event.key == pygame.K_F6:  # Decrease contrast
                        self.camera.adjust_ir_settings(contrast=1.4)
                    elif event.key == pygame.K_F7:  # Increase exposure
                        self.camera.adjust_ir_settings(exposure_time=15000)
                    elif event.key == pygame.K_F8:  # Decrease exposure
                        self.camera.adjust_ir_settings(exposure_time=8000)
                    # Debug and threshold shortcuts
                    elif event.key == pygame.K_F9:  # Toggle debug mode
                        self.processor.debug_mode = not self.processor.debug_mode
                        print(f"🔍 Debug mode: {'ON' if self.processor.debug_mode else 'OFF'}")
                    elif event.key == pygame.K_F10:  # Lower confidence threshold
                        self.processor.confidence_threshold = max(0.1, self.processor.confidence_threshold - 0.1)
                        print(f"📉 Confidence threshold: {self.processor.confidence_threshold:.1f}")
                    elif event.key == pygame.K_F11:  # Raise confidence threshold
                        self.processor.confidence_threshold = min(0.9, self.processor.confidence_threshold + 0.1)
                        print(f"📈 Confidence threshold: {self.processor.confidence_threshold:.1f}")
                    elif event.key == pygame.K_F12:  # Toggle RGB mode
                        self.processor.force_rgb_mode = not self.processor.force_rgb_mode
                        print(f"🎨 RGB mode: {'ON' if self.processor.force_rgb_mode else 'OFF'}")
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_button_click(event.pos)
            
            # TODO: Consider threaded producer/consumer queue for decoupling capture from inference (better performance on Pi)
            # Process camera frame based on mode
            if self.capture_only_mode:
                # Capture mode: Just display live feed without processing
                frame = self.camera.capture_frame()
                if frame is not None:
                    # Show live preview in OpenCV window (Note: Mixing Pygame and OpenCV can cause input issues)
                    cv2.imshow('Training Data Preview - Press C to capture', frame)
                    if cv2.waitKey(1) == ord('c'):
                        self.capture_training_image()
            
            elif self.processing_enabled:
                # AI processing mode - high resolution display with ML processing
                display_frame = self.camera.capture_frame()  # 1024x1024 for display
                if display_frame is not None:
                    start_time = time.time()
                    results = self.processor.process_frame(display_frame)  # Internally uses 320x320 for ML
                    processing_time = time.time() - start_time
                    
                    # Update stats
                    self.stats['frames_processed'] += 1
                    self.stats['detections_found'] += len(results)
                    self.stats['processing_time'] = processing_time * 1000  # ms
                    
                    # Store latest results
                    if results:
                        self.last_results = results
                    
                    # Display frame with results on HIGH-RESOLUTION display
                    if results:
                        draw_detection_results(display_frame, results)
                    
                    # Show crystal clear 1024x1024 display (10x better than 320x320)
                    cv2.imshow('OCR Detection Test (1024x1024 Display)', display_frame)
                    if cv2.waitKey(1) == ord('q'):
                        self.running = False
            
            else:
                # Idle mode: Just show live preview without processing
                display_frame = self.camera.capture_frame()  # 1024x1024 for display
                if display_frame is not None:
                    # Show live camera feed without processing
                    cv2.imshow('OCR Detection Test (1024x1024 Display) - PAUSED', display_frame)
                    if cv2.waitKey(1) == ord('q'):
                        self.running = False
            
            # Draw UI
            self.draw_ui()
            pygame.display.flip()
            clock.tick(30)  # 30 FPS
        
        self.cleanup()
    
    def cleanup(self):
        """Clean up all resources"""
        print("🧹 Cleaning up...")
        self.camera.cleanup()
        self.processor.cleanup()
        pygame.quit()
        cv2.destroyAllWindows()

def main():
    """Main entry point"""
    print("🚀 Starting OCR Edge AI Test")
    print("📋 This will test SIM card detection + OCR extraction pipeline")
    
    try:
        app = OCRTestUI()
        app.run()
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Application error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("👋 OCR Test completed")

if __name__ == "__main__":
    main()