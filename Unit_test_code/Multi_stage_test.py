#!/usr/bin/env python3
"""
Multi-Stage Edge AI Test - SIM Card Detection + OCR Extraction
Based on Edge Impulse two-stage processing pipeline
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
PREVIEW_WIDTH = 1280
PREVIEW_HEIGHT = 720
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
ocr_runner = None

def signal_handler(sig, frame):
    """Clean shutdown on SIGINT"""
    global show_camera, detection_runner, ocr_runner
    print('🛑 Interrupted - Shutting down...')
    show_camera = False
    if detection_runner:
        detection_runner.stop()
    if ocr_runner:
        ocr_runner.stop()
    pygame.quit()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def draw_detection_results(frame, detections):
    """
    Draw bounding boxes and labels for detected SIM cards with OCR results
    Following the multi_stage.py pattern
    """
    for detection in detections:
        bbox = detection['bbox']
        x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
        
        # Detection info
        sim_type = detection.get('label', 'Unknown')
        confidence = detection.get('confidence', 0.0)
        ocr_result = detection.get('ocr_digits', 'No OCR')
        
        # Choose color based on SIM type
        if 'MTN' in sim_type.upper():
            color = (0, 255, 255)  # Yellow for MTN
        elif 'VODACOM' in sim_type.upper():
            color = (0, 0, 255)    # Red for Vodacom  
        else:
            color = (255, 0, 0)    # Blue for unknown
        
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
        
        # Create comprehensive label
        label = f"{sim_type} ({confidence:.2f})"
        ocr_label = f"OCR: {ocr_result}"
        
        # Calculate text size and position
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
        (ocr_w, ocr_h), _ = cv2.getTextSize(ocr_label, font, font_scale, thickness)
        
        # Position labels above bounding box
        label_y = y - 40 if y > 50 else y + h + 30
        ocr_y = label_y + 25
        
        # Draw label backgrounds
        cv2.rectangle(frame, (x, label_y - text_h - 5), (x + max(text_w, ocr_w) + 10, ocr_y + ocr_h + 5), color, -1)
        
        # Draw labels
        cv2.putText(frame, label, (x + 5, label_y - 5), font, font_scale, (255, 255, 255), thickness)
        cv2.putText(frame, ocr_label, (x + 5, ocr_y - 5), font, font_scale, (255, 255, 255), thickness)

class MultiStageProcessor:
    """
    Two-stage Edge Impulse processor following multi_stage.py pattern:
    Stage 1: SIM Card Detection
    Stage 2: OCR Extraction from detected regions
    """
    
    def __init__(self):
        self.detection_runner = None
        self.ocr_runner = None
        self.detection_labels = []
        self.ocr_labels = []
        self.ocr_input_width = 0
        self.ocr_input_height = 0
        self.initialized = False
        
        # Detection filtering parameters
        self.confidence_threshold = 0.7  # Higher threshold to reduce false positives
        self.nms_threshold = 0.4  # IoU threshold for NMS
        self.skip_background = True  # Skip background detections for OCR
        
    def initialize_models(self):
        """Initialize both Edge Impulse models"""
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Model paths
            detection_model_path = os.path.join(script_dir, "models", "simcard_detection.eim")
            ocr_model_path = os.path.join(script_dir, "models", "ocr_extraction.eim")
            
            # Verify models exist
            if not os.path.exists(detection_model_path):
                raise FileNotFoundError(f"SIM detection model not found: {detection_model_path}")
            if not os.path.exists(ocr_model_path):
                raise FileNotFoundError(f"OCR extraction model not found: {ocr_model_path}")
            
            print(f"📁 Loading detection model: {os.path.basename(detection_model_path)}")
            self.detection_runner = ImageImpulseRunner(detection_model_path)
            detection_model_info = self.detection_runner.init()
            
            print(f"📁 Loading OCR model: {os.path.basename(ocr_model_path)}")
            self.ocr_runner = ImageImpulseRunner(ocr_model_path)
            ocr_model_info = self.ocr_runner.init()
            
            # Extract model information
            print(f'✅ Detection model loaded: "{detection_model_info["project"]["owner"]} / {detection_model_info["project"]["name"]}"')
            self.detection_labels = detection_model_info['model_parameters']['labels']
            print(f"🏷️ Detection labels: {self.detection_labels}")
            
            print(f'✅ OCR model loaded: "{ocr_model_info["project"]["owner"]} / {ocr_model_info["project"]["name"]}"')
            self.ocr_labels = ocr_model_info['model_parameters']['labels']
            self.ocr_input_width = ocr_model_info['model_parameters']['image_input_width']
            self.ocr_input_height = ocr_model_info['model_parameters']['image_input_height']
            print(f"🏷️ OCR labels: {self.ocr_labels}")
            print(f"📐 OCR input size: {self.ocr_input_width}x{self.ocr_input_height}")
            
            self.initialized = True
            return True
            
        except Exception as e:
            print(f"❌ Model initialization failed: {e}")
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
                continue
                
            # Skip background detections for OCR processing
            if self.skip_background and 'background' in current['label'].lower():
                print(f"🚫 Skipping background detection: {current['label']} ({current['value']:.2f})")
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
    
    def process_frame(self, frame):
        """
        Two-stage processing following multi_stage.py pattern:
        1. Detect SIM cards
        2. Extract OCR from detected regions
        """
        if not self.initialized:
            return []
        
        results = []
        
        try:
            # Stage 1: SIM Card Detection
            print("🔍 Running SIM card detection...")
            detection_features, processed_img = self.detection_runner.get_features_from_image(frame)
            detection_result = self.detection_runner.classify(detection_features)
            
            processing_time = detection_result['timing']['dsp'] + detection_result['timing']['classification']
            print(f'📊 Detection: Found {len(detection_result.get("result", {}).get("bounding_boxes", []))} bounding boxes ({processing_time} ms.)')
            
            # Apply NMS filtering to remove duplicate detections
            raw_detections = detection_result.get("result", {}).get("bounding_boxes", [])
            if raw_detections:
                print(f"🔍 Raw detections: {len(raw_detections)}")
                for i, bbox in enumerate(raw_detections):
                    print(f"  {i+1}. {bbox['label']} ({bbox['value']:.2f}): x={bbox['x']} y={bbox['y']} w={bbox['width']} h={bbox['height']}")
                
                # Apply NMS filtering
                filtered_detections = self.apply_nms(raw_detections)
                
                # Process each filtered SIM card detection
                for bbox in filtered_detections:
                    sim_detection = {
                        'label': bbox['label'],
                        'confidence': bbox['value'],
                        'bbox': bbox,
                        'ocr_digits': 'Processing...'
                    }
                    
                    print(f'📋 Processing: {bbox["label"]} (confidence: {bbox["value"]:.2f}): x={bbox["x"]} y={bbox["y"]} w={bbox["width"]} h={bbox["height"]}')
                    
                    # Stage 2: OCR Extraction on high-confidence SIM card detections
                    if bbox['value'] >= self.confidence_threshold:
                        ocr_result = self.extract_ocr_from_detection(frame, bbox)
                        sim_detection['ocr_digits'] = ocr_result
                        print(f'🔤 OCR Result: {ocr_result}')
                    else:
                        sim_detection['ocr_digits'] = f'Low confidence ({bbox["value"]:.2f})'
                        print(f'⚠️ Skipped OCR: Low confidence ({bbox["value"]:.2f})')
                    
                    results.append(sim_detection)
                    
        except Exception as e:
            print(f"❌ Processing error: {e}")
            
        return results
    
    def extract_ocr_from_detection(self, frame, bbox):
        """Extract OCR digits from detected SIM card region"""
        try:
            # Crop the detected region
            x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
            cropped_sim = frame[y:y+h, x:x+w]
            
            if cropped_sim.size == 0:
                return "Invalid crop"
            
            # Debug: Save cropped image to see what OCR model receives
            timestamp = int(time.time() * 1000)
            debug_filename = f"debug_crop_{timestamp}.jpg"
            cv2.imwrite(debug_filename, cropped_sim)
            print(f"🐛 Saved debug crop: {debug_filename}")
            
            # Resize to OCR model input size
            resized_sim = cv2.resize(cropped_sim, (self.ocr_input_width, self.ocr_input_height))
            
            # Run OCR model
            ocr_features, processed_crop = self.ocr_runner.get_features_from_image(resized_sim)
            ocr_result = self.ocr_runner.classify(ocr_features)
            
            ocr_time = ocr_result['timing']['dsp'] + ocr_result['timing']['classification']
            print(f'📝 OCR processing time: {ocr_time} ms.')
            
            # DEBUG: Print entire OCR result structure
            print(f"🐛 Full OCR result structure: {ocr_result}")
            print(f"🐛 OCR result keys: {ocr_result.get('result', {}).keys()}")
            
            # Parse OCR results - Handle multiple possible formats
            result_data = ocr_result.get("result", {})
            
            if "classification" in result_data:
                # Handle classification format (your MNIST model)
                classifications = result_data["classification"]
                print(f'🐛 Classifications found: {classifications}')
                
                top_score = 0
                top_digit = 'unknown'
                
                digit_results = []
                for label in self.ocr_labels:
                    score = classifications.get(label, 0)
                    digit_results.append(f'{label}: {score:.2f}')
                    if score > top_score:
                        top_score = score
                        top_digit = label
                
                print(f'🎯 OCR classifications: {", ".join(digit_results)}')
                print(f'🏆 Top OCR result: {top_digit} (confidence: {top_score:.2f})')
                
                return f"{top_digit} ({top_score:.2f})"
                
            elif "bounding_boxes" in result_data:
                # Handle object detection format (if OCR model is detection-based)
                boxes = result_data["bounding_boxes"]
                if boxes:
                    best_box = max(boxes, key=lambda x: x.get('value', 0))
                    return f"{best_box.get('label', 'unknown')} ({best_box.get('value', 0):.2f})"
                else:
                    return "No detections"
                    
            else:
                # Handle unknown format
                print(f"🐛 Unknown OCR result format. Available keys: {result_data.keys()}")
                return f"Unknown format: {list(result_data.keys())}"
                
        except Exception as e:
            print(f"❌ OCR extraction error: {e}")
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)}"
    
    def cleanup(self):
        """Clean shutdown of models"""
        if self.detection_runner:
            self.detection_runner.stop()
        if self.ocr_runner:
            self.ocr_runner.stop()

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
        """Initialize Raspberry Pi camera"""
        try:
            self.camera = Picamera2()
            config = self.camera.create_preview_configuration(
                main={"size": (PREVIEW_WIDTH, PREVIEW_HEIGHT), "format": "RGB888"}
            )
            self.camera.configure(config)
            self.camera.start()
            time.sleep(2)  # Camera warm-up
            print("✅ Picamera2 initialized successfully")
            return True
        except Exception as e:
            print(f"❌ Picamera2 initialization failed: {e}")
            return False
    
    def init_opencv_camera(self):
        """Initialize OpenCV camera for development"""
        try:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                return False
            
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, PREVIEW_WIDTH)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, PREVIEW_HEIGHT)
            print("✅ OpenCV camera initialized successfully")
            return True
        except Exception as e:
            print(f"❌ OpenCV camera initialization failed: {e}")
            return False
    
    def capture_frame(self):
        """Capture a frame from the camera"""
        try:
            if RASPBERRY_PI_MODE and self.camera:
                frame = self.camera.capture_array()
                return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            elif self.camera:
                ret, frame = self.camera.read()
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

class MultiStageUI:
    """PyGame UI for multi-stage Edge AI testing"""
    
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Multi-Stage Edge AI Test - SIM Detection + OCR")
        
        self.font = pygame.font.Font(None, 24)
        self.title_font = pygame.font.Font(None, 36)
        
        self.camera = CameraController()
        self.processor = MultiStageProcessor()
        
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
        
        # Organized training data structure
        self.setup_training_directories()
        self.classification_counts = {
            'MTN': 0,
            'Vodacom': 0,
            'background': 0,
            'raw': 0
        }
    
    def initialize(self):
        """Initialize all components"""
        print("🚀 Initializing Multi-Stage Test UI...")
        
        if not self.camera.init_camera():
            print("❌ Camera initialization failed")
            return False
        
        if not self.capture_only_mode:
            if not self.processor.initialize_models():
                print("❌ Model initialization failed")
                return False
        else:
            print("📷 Capture-only mode: Skipping AI model initialization")
        
        print("✅ All components initialized successfully")
        return True
    
    def setup_training_directories(self):
        """Create organized folder structure for training data"""
        base_dir = "training_data"
        self.training_dirs = {
            'MTN': os.path.join(base_dir, "MTN"),
            'Vodacom': os.path.join(base_dir, "Vodacom"),
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
        print(f"📊 Counts - MTN: {self.classification_counts['MTN']}, Vodacom: {self.classification_counts['Vodacom']}, Background: {self.classification_counts['background']}, Raw: {self.classification_counts['raw']}")
        
        return filepath
    
    def setup_training_directories(self):
        """Create organized folder structure for training data"""
        base_dir = "training_data"
        self.training_dirs = {
            'MTN': os.path.join(base_dir, "MTN"),
            'Vodacom': os.path.join(base_dir, "Vodacom"),
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
        print(f"📊 Counts - MTN: {self.classification_counts['MTN']}, Vodacom: {self.classification_counts['Vodacom']}, Background: {self.classification_counts['background']}, Raw: {self.classification_counts['raw']}")
        
        return filepath
    
    def draw_ui(self):
        """Draw the main UI"""
        self.screen.fill(WHITE)
        
        # Title with mode indicator
        mode_text = " - CAPTURE MODE" if self.capture_only_mode else " - AI PROCESSING MODE"
        title_text = self.title_font.render(f"Multi-Stage Edge AI Test{mode_text}", True, BLACK)
        self.screen.blit(title_text, (MARGIN, MARGIN))
        
        # Status info
        status_y = 60
        status_texts = [
            f"Camera: {'✅ Active' if self.camera.camera else '❌ Failed'}",
            f"Mode: {'📷 Capture Only' if self.capture_only_mode else '🤖 AI Processing'}",
            f"Models: {'⏭️ Skipped' if self.capture_only_mode else ('✅ Loaded' if self.processor.initialized else '❌ Failed')}",
            f"Processing: {'⏸️ Disabled (Capture Mode)' if self.capture_only_mode else ('✅ Enabled' if self.processing_enabled else '⏸️ Disabled')}",
            f"Total Images: {self.stats['images_captured']}",
            f"MTN: {self.classification_counts['MTN']} | Vodacom: {self.classification_counts['Vodacom']} | Background: {self.classification_counts['background']} | Raw: {self.classification_counts['raw']}",
            f"Frames Processed: {self.stats['frames_processed']}, Detections: {self.stats['detections_found']}"
        ]
        
        for i, text in enumerate(status_texts):
            color = PURPLE if "Capture" in text else BLACK
            if "MTN:" in text:  # Color code the classification counts
                color = DARK_GRAY
            rendered = self.font.render(text, True, color)
            self.screen.blit(rendered, (MARGIN, status_y + i * 25))
        
        # Control buttons - different layout for capture mode
        button_y = 240
        if self.capture_only_mode:
            self.draw_button("📷 Background (B)", MARGIN, button_y, GREEN)
            self.draw_button("📁 MTN (M)", MARGIN + BUTTON_WIDTH + 10, button_y, BLUE)
            self.draw_button("📱 Vodacom (V)", MARGIN + 2 * (BUTTON_WIDTH + 10), button_y, RED)
            self.draw_button("🤖 AI Mode", MARGIN + 3 * (BUTTON_WIDTH + 10), button_y, ORANGE)
            
            # Second row of buttons
            button_y2 = button_y + BUTTON_HEIGHT + 10
            self.draw_button("📄 Raw (C)", MARGIN, button_y2, GRAY)
            self.draw_button("📁 Batch BG (10)", MARGIN + BUTTON_WIDTH + 10, button_y2, DARK_GRAY)
            self.draw_button("🔄 Reset Stats", MARGIN + 2 * (BUTTON_WIDTH + 10), button_y2, PURPLE)
            self.draw_button("Quit", MARGIN + 3 * (BUTTON_WIDTH + 10), button_y2, DARK_GRAY)
        else:
            self.draw_button("Toggle Processing", MARGIN, button_y, GREEN if self.processing_enabled else RED)
            self.draw_button("Capture & Save", MARGIN + BUTTON_WIDTH + 10, button_y, BLUE)
            self.draw_button("📷 Capture Mode", MARGIN + 2 * (BUTTON_WIDTH + 10), button_y, PURPLE)
            self.draw_button("Quit", MARGIN + 3 * (BUTTON_WIDTH + 10), button_y, DARK_GRAY)
        
        # Results display
        results_y = 280
        
        # Add keyboard shortcuts help for capture mode
        if self.capture_only_mode:
            help_text1 = "Single: M=MTN, V=Vodacom, B=Background, C=Raw, R=Reset, Q=Quit"
            help_text2 = "Batch (10x): 1=Background, 2=MTN, 3=Vodacom, 4=Raw"
            help_rendered1 = self.font.render(help_text1, True, PURPLE)
            help_rendered2 = self.font.render(help_text2, True, PURPLE)
            self.screen.blit(help_rendered1, (MARGIN, results_y))
            self.screen.blit(help_rendered2, (MARGIN, results_y + 20))
            results_y += 45
        
        results_title = self.font.render("Latest Detection Results:", True, BLACK)
        self.screen.blit(results_title, (MARGIN, results_y))
        
        for i, detection in enumerate(self.last_results[:5]):  # Show last 5 results
            result_text = f"{detection.get('label', 'Unknown')}: {detection.get('confidence', 0):.2f}, OCR: {detection.get('ocr_digits', 'None')}"
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
        button_y = 240
        button_y2 = button_y + BUTTON_HEIGHT + 10
        
        # First row buttons
        buttons_row1 = [
            pygame.Rect(MARGIN, button_y, BUTTON_WIDTH, BUTTON_HEIGHT),  # Button 1
            pygame.Rect(MARGIN + BUTTON_WIDTH + 10, button_y, BUTTON_WIDTH, BUTTON_HEIGHT),  # Button 2
            pygame.Rect(MARGIN + 2 * (BUTTON_WIDTH + 10), button_y, BUTTON_WIDTH, BUTTON_HEIGHT),  # Button 3
            pygame.Rect(MARGIN + 3 * (BUTTON_WIDTH + 10), button_y, BUTTON_WIDTH, BUTTON_HEIGHT),  # Button 4
        ]
        
        # Second row buttons (capture mode only)
        buttons_row2 = [
            pygame.Rect(MARGIN, button_y2, BUTTON_WIDTH, BUTTON_HEIGHT),  # Raw
            pygame.Rect(MARGIN + BUTTON_WIDTH + 10, button_y2, BUTTON_WIDTH, BUTTON_HEIGHT),  # Batch BG
            pygame.Rect(MARGIN + 2 * (BUTTON_WIDTH + 10), button_y2, BUTTON_WIDTH, BUTTON_HEIGHT),  # Reset
            pygame.Rect(MARGIN + 3 * (BUTTON_WIDTH + 10), button_y2, BUTTON_WIDTH, BUTTON_HEIGHT),  # Quit
        ]
        
        if self.capture_only_mode:
            # First row - capture mode
            if buttons_row1[0].collidepoint(pos):  # Background
                frame = self.camera.capture_frame()
                if frame is not None:
                    self.save_classified_image(frame, 'background')
            elif buttons_row1[1].collidepoint(pos):  # MTN
                frame = self.camera.capture_frame()
                if frame is not None:
                    self.save_classified_image(frame, 'MTN')
            elif buttons_row1[2].collidepoint(pos):  # Vodacom
                frame = self.camera.capture_frame()
                if frame is not None:
                    self.save_classified_image(frame, 'Vodacom')
            elif buttons_row1[3].collidepoint(pos):  # Switch to AI
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
                filename = f"multistage_detection_{timestamp}.jpg"
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
        
        print("🎮 Multi-Stage Test UI started - Press SPACE to toggle processing, ESC to quit")
        
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
                    elif event.key == pygame.K_s:
                        self.capture_and_save()
                    # Classification capture shortcuts (only in capture mode)
                    elif self.capture_only_mode:
                        if event.key == pygame.K_m:  # MTN
                            frame = self.camera.capture_frame()
                            if frame is not None:
                                self.save_classified_image(frame, 'MTN')
                        elif event.key == pygame.K_v:  # Vodacom
                            frame = self.camera.capture_frame()
                            if frame is not None:
                                self.save_classified_image(frame, 'Vodacom')
                        elif event.key == pygame.K_b:  # Background
                            frame = self.camera.capture_frame()
                            if frame is not None:
                                self.save_classified_image(frame, 'background')
                        elif event.key == pygame.K_c:  # Raw (C for capture)
                            frame = self.camera.capture_frame()
                            if frame is not None:
                                self.save_classified_image(frame, 'raw')
                        elif event.key == pygame.K_r:  # Reset stats
                            self.reset_stats()
                        elif event.key == pygame.K_q:  # Quit
                            self.running = False
                        # Batch capture shortcuts (Shift + key)
                        elif event.key == pygame.K_1:  # Batch background (10 images)
                            self.batch_capture_classified('background', 10)
                        elif event.key == pygame.K_2:  # Batch MTN (10 images)  
                            self.batch_capture_classified('MTN', 10)
                        elif event.key == pygame.K_3:  # Batch Vodacom (10 images)
                            self.batch_capture_classified('Vodacom', 10)
                        elif event.key == pygame.K_4:  # Batch Raw (10 images)
                            self.batch_capture_classified('raw', 10)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_button_click(event.pos)
            
            # Process camera frame based on mode
            if self.capture_only_mode:
                # Capture mode: Just display live feed without processing
                frame = self.camera.capture_frame()
                if frame is not None:
                    # Show live preview in OpenCV window
                    cv2.imshow('Training Data Preview - Press C to capture', frame)
                    if cv2.waitKey(1) == ord('c'):
                        self.capture_training_image()
            
            elif self.processing_enabled:
                # AI processing mode
                frame = self.camera.capture_frame()
                if frame is not None:
                    start_time = time.time()
                    results = self.processor.process_frame(frame)
                    processing_time = time.time() - start_time
                    
                    # Update stats
                    self.stats['frames_processed'] += 1
                    self.stats['detections_found'] += len(results)
                    self.stats['processing_time'] = processing_time * 1000  # ms
                    
                    # Store latest results
                    if results:
                        self.last_results = results
                    
                    # Display frame with results
                    if results:
                        draw_detection_results(frame, results)
                    
                    # Show in OpenCV window
                    cv2.imshow('Multi-Stage Detection', frame)
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
    print("🚀 Starting Multi-Stage Edge AI Test")
    print("📋 This will test SIM card detection + OCR extraction pipeline")
    
    try:
        app = MultiStageUI()
        app.run()
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Application error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("👋 Multi-Stage Test completed")

if __name__ == "__main__":
    main()