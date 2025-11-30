# PI Imaging - System Architecture Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Component Architecture](#component-architecture)
3. [Data Flow](#data-flow)
4. [Module Interactions](#module-interactions)
5. [Communication Protocols](#communication-protocols)
6. [Deployment Topology](#deployment-topology)

---

## System Overview

PI Imaging is a distributed edge AI system that automates SIM card detection, extraction, and OCR processing. The system operates as a multi-tier architecture with hardware controllers, edge intelligence, cloud storage, and workflow automation.

### Key Characteristics
- **Edge-First Design**: ML inference runs locally on Raspberry Pi (no cloud latency)
- **Hardware Integration**: ESP32 motor controller + IR camera for automated dispensing
- **Real-time Streaming**: WebSocket-based live video feed with performance metrics
- **Distributed Processing**: AWS IoT Core → Lambda → n8n workflow orchestration
- **High Throughput**: 90+ SIM cards processed per minute (6x improvement over manual)

---

## Component Architecture

### Layer 1: Hardware Controller (ESP32)

**Purpose**: Motor control and SIM card dispensing
**Language**: Arduino/C++

```
┌─────────────────────────────────┐
│     ESP32 DevKit Board          │
├─────────────────────────────────┤
│ GPIO 19 → Stepper STEP          │
│ GPIO 18 → Stepper DIR           │
│ GPIO 23 → Stepper ENABLE        │
│ GPIO 5  → Pusher Servo (PWM)    │
│ GPIO 23 → Position Servo (PWM)  │
├─────────────────────────────────┤
│ Serial Interface (115200 baud)  │
│ Commands: DISPENSE, MOVE, TEST  │
└─────────────────────────────────┘
         ↓
    12V Stepper Driver
    2× Servo Motors
    Conveyor Belt
```

**Communication**: Serial UART (115200 baud) ↔ Raspberry Pi

**Commands Available**:
- `DISPENSE <1-3>` - Dispense 3 cards at position
- `DISPENSE_ALL` - Sequence all 9 cards
- `MOVE <steps> <direction>` - Conveyor control
- `RESET` - Home position
- `TEST` - Full system test

**File**: `ESP32_Motor_Controller/ESP32_Motor_Controller.ino`

---

### Layer 2: Edge Intelligence (Raspberry Pi 4)

The Raspberry Pi 4 hosts the main system intelligence with dual-purpose streaming architecture.

#### 2A. Camera System (`camera_system.py`)

**Purpose**: High-resolution capture + Edge Impulse inference + AWS integration

**Key Classes**:
```
CameraSystem
├── Hardware Layer
│   ├── Picamera2 (2560×1440 resolution)
│   ├── ESP32SerialController
│   └── EdgeImpulseRunner
├── Processing Layer
│   ├── Frame Capture (high-res, OCR-optimized)
│   ├── ML Detection (MobileNetV2 SSD FPN-Lite)
│   ├── Image Cropping (per-card extraction)
│   └── S3 Upload Queue
├── Integration Layer
│   ├── AWS S3 Uploader
│   ├── AWS IoT MQTT Publisher
│   └── Webhook Dispatcher
└── Monitoring Layer
    ├── Performance Metrics
    ├── Error Recovery
    └── Circuit Breaker Pattern
```

**Processing Pipeline**:
```
1. CAPTURE: Picamera2 (2560×1440 @ 30fps)
   ↓
2. STREAM: Dual-stream architecture
   ├─ Stream Buffer (1280×720 @ 15fps) → WebSocket
   └─ Inference Buffer (320×320) → Edge Impulse
   ↓
3. DETECT: MobileNetV2 SSD FPN-Lite model
   - Inference time: 372ms/frame
   - mAP: 95% F1-score
   - Detects up to 10 objects/frame
   ↓
4. EXTRACT: Crop detected SIM cards
   - Individual card isolation
   - Border padding for OCR
   ↓
5. UPLOAD: Parallel S3 upload
   - Retryable queue (max 5 items)
   - 2 concurrent upload threads
   ↓
6. NOTIFY: AWS IoT MQTT
   - Message format: JSON metadata
   - Topic: pi-imaging/detections
   - QoS: 1 (at-least-once delivery)
```

**Key Methods**:
- `capture_and_process_optimized()` - Main processing loop
- `get_stream_frame()` - Stream buffer access (WebSocket)
- `detect_simcards()` - Edge Impulse inference
- `crop_and_upload()` - Image extraction & S3
- `publish_detections()` - AWS IoT notification

**File**: `camera_system.py` (2591 lines)

---

#### 2B. Flask Application (`app.py`)

**Purpose**: REST API + WebSocket server + Real-time dashboard

**Architecture**:
```
Flask App (Port 5000)
├── REST API Routes
│   ├── /api/start → system.start_system()
│   ├── /api/stop → system.stop_system()
│   ├── /api/status → get_status()
│   ├── /api/performance → get_performance_metrics()
│   ├── /api/health → health_check()
│   └── /api/settings → configuration
├── WebSocket Events (Socket.IO)
│   ├── connect → stream_thread.start()
│   ├── frame_update → base64 JPEG (15 FPS)
│   ├── status_update → system state
│   ├── performance_update → real-time metrics
│   ├── log_update → streaming logs
│   └── error_update → error alerts
└── Static Assets
    ├── /assets/* → React build files
    ├── / → index.html (React Router)
    └── /diagnostics → Troubleshooting page
```

**Request/Response Cycle**:
```
Client Request
    ↓
Flask Route Handler
    ↓
Camera System Call
    ↓
Status Callback (async)
    ↓
SocketIO Emit to All Clients
    ↓
Response JSON
```

**Key Endpoints**:
| Endpoint | Method | Purpose | Response |
|----------|--------|---------|----------|
| `/api/start` | POST | Start detection system | `{run_id, status}` |
| `/api/stop` | POST | Stop system | `{final_count, summary}` |
| `/api/status` | GET | System state | Full system_state object |
| `/api/health` | GET | Diagnostics | Camera, MQTT, S3, connectivity |
| `/api/performance` | GET | Metrics | Throughput, latency, accuracy |
| `/api/camera/feed` | GET | Video stream | MJPEG stream (95% quality) |

**Status Callback System**:
The `status_callback()` function handles asynchronous updates from camera_system:
- Emits WebSocket events in real-time
- Logs to console with timestamps
- Updates global `system_state` dictionary
- Handles 4 callback types: log_update, error_update, exception_details, system_stopped

**File**: `app.py` (750+ lines)

---

### Layer 3: Frontend Dashboard (React + TypeScript)

**Purpose**: Real-time monitoring, control, and visualization

**Structure**:
```
frontend/
├── src/
│   ├── App.tsx (Root component)
│   ├── main.tsx (Entry point)
│   ├── pages/
│   │   └── Index.tsx (Main dashboard)
│   ├── components/
│   │   ├── SimProcessing/ (Detection display)
│   │   └── ui/ (Shadcn components)
│   ├── hooks/
│   │   ├── useSystemSocket.ts (WebSocket handler)
│   │   └── use-mobile.tsx (Responsive)
│   └── lib/ (Utilities)
└── dist/ (Build output)
```

**WebSocket Events Handled**:
```
Connection Events:
  - connection_confirmed
  - frame_update → Display live video
  - status → Update system state
  
Performance Events:
  - performance_update → Metrics graph
  - performance_metrics → Detailed stats
  
Error Events:
  - error_update → Alert user
  - errors_cleared → Clear notifications
  
System Events:
  - system_started → Enable stop button
  - system_stopped → Show summary
  - system_paused/resumed → Control UI
```

**Key Components**:
- **useSystemSocket Hook**: Manages WebSocket lifecycle & data
- **SimProcessing Component**: Displays detected cards with bounding boxes
- **Performance Dashboard**: Real-time graphs & metrics
- **Live Feed**: MJPEG stream + overlay data

**Build Process**:
```bash
npm install
npm run build     # Output: frontend/dist/
```

**File**: `frontend/src/**` (React + Vite)

---

## Data Flow

### 1. Real-Time Detection Flow

```
ESP32 Dispenses 9 Cards
    ↓
Raspberry Pi Captures Frame (2560×1440)
    ↓
Dual-Stream Processing:
├─ Stream Path (1280×720, 15fps)
│  └─→ WebSocket → Browser Display
└─ Inference Path (320×320)
    └─→ Edge Impulse MobileNetV2 SSD
        └─→ Detects up to 10 SIM cards
            └─→ Bounding boxes + confidence scores
                ↓
            For each detected card:
            ├─ Crop ROI from original frame
            ├─ Add border padding
            ├─ Encode as JPEG (95% quality)
            └─ Queue for upload
                ↓
            S3 Upload (async, 2 threads)
            └─→ Crop stored as:
                s3://bucket/run_id/detection_id/crop.jpg
                ↓
            AWS IoT MQTT Publish
            └─→ {
                  "run_id": "...",
                  "detection_id": "...",
                  "card_count": 9,
                  "confidence": [0.95, 0.93, ...],
                  "s3_keys": ["crop_1.jpg", "crop_2.jpg", ...]
                }
                ↓
            Lambda Function Triggered
            └─→ Forwards to n8n webhook
                ↓
            n8n OCR Workflow
            └─→ Download image
                Extract text (ICCID, phone)
                Validate format
                Store in Airtable
                ↓
            Power BI Ingestion
            └─→ Real-time analytics dashboard
```

**Timing**:
- Frame capture: 33ms (30fps)
- ML inference: 372ms
- S3 upload: 200-500ms per image
- Total batch time: ~2 seconds for 9 cards
- **Throughput**: 90+ cards/minute

---

### 2. System State Management

```
Global system_state Dictionary:
{
  "running": bool,
  "paused": bool,
  "current_run_id": str,
  "service_provider": str,
  "image_count": int,
  "total_runs": int,
  "last_image_time": datetime,
  "error_message": str,
  "start_time": datetime,
  "performance": {
    "avg_capture_time": float,
    "avg_upload_time": float,
    "images_per_minute": float,
    "detection_accuracy": float
  }
}
```

**State Transitions**:
```
IDLE
  │
  └─→ /api/start (POST service_provider)
      │
      └─→ RUNNING
          ├─ Capture frames every 2 seconds
          ├─ Detect SIM cards (ML inference)
          ├─ Upload crops (S3)
          ├─ Publish metadata (MQTT)
          │
          └─→ /api/stop (POST)
              │
              └─→ STOPPED (emit performance summary)
          │
          └─→ /api/pause (POST)
              │
              └─→ PAUSED
                  │
                  └─→ /api/resume (POST) → RUNNING
```

---

## Module Interactions

### ESP32 ↔ Raspberry Pi

**Serial Protocol**:
```
Request Format:  "COMMAND <arg1> <arg2>\n"
Response Format: "OK\n" or "ERROR: <msg>\n"

Examples:
  → "DISPENSE 1\n"
  ← "OK\n" (3 cards dispensed at 0°)
  
  → "MOVE 980 1\n"
  ← "OK\n" (Conveyor moved 10cm)
  
  → "TEST\n"
  ← "OK\n" (Full 9-card test sequence)
```

**Flow**:
```
camera_system.start_system()
  ↓
esp32_controller.dispense_all()
  ├─ ser.write(b"DISPENSE 1\n")
  ├─ Wait for response
  ├─ ser.write(b"DISPENSE 2\n")
  ├─ Wait for response
  ├─ ser.write(b"DISPENSE 3\n")
  ├─ Wait for response
  └─ Return to main capture loop
```

---

### Raspberry Pi ↔ AWS IoT Core

**MQTT Connection**:
```
Endpoint: {IOT_ENDPOINT} (eu-north-1)
Client ID: raspberrypi-alpha
Auth: X.509 Certificates
  - Cert: certs/device.cert.pem
  - Key: certs/device.private.key
  - CA: certs/AmazonRootCA1.pem
```

**Message Publishing**:
```python
mqtt_connection.publish(
  topic="pi-imaging/detections",
  payload=json.dumps({
    "run_id": "abc123",
    "timestamp": "2025-11-29T10:30:45Z",
    "frame_number": 42,
    "detected_cards": 9,
    "confidence_scores": [0.95, 0.93, ...],
    "s3_crops": [
      "s3://bucket/abc123/det_001/crop.jpg",
      "s3://bucket/abc123/det_002/crop.jpg",
      ...
    ]
  }),
  qos=1  # At-least-once delivery
)
```

**IoT Rule** (AWS Console):
```
SELECT * FROM 'pi-imaging/detections'
└─ Action: Lambda (pi_imaging_to_n8n)
    └─ Invokes: n8n webhook → OCR processing
```

---

### Raspberry Pi ↔ AWS S3

**Authentication**: IAM credentials from environment variables
```
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_REGION = "eu-north-1"
S3_BUCKET_NAME = "ocrstorage4d"
```

**Upload Flow**:
```
Detected Card Crop
  ↓
boto3.client('s3').put_object(
  Bucket=S3_BUCKET_NAME,
  Key=f"run_{run_id}/detection_{detection_id}/crop.jpg",
  Body=image_bytes,
  ContentType='image/jpeg'
)
  ↓
Upload Queue Status Tracked
  ├─ Success: Remove from queue
  ├─ Failure: Retry (max 3 attempts)
  └─ Queue Full: Circuit breaker (stop capture)
```

**Object Structure**:
```
s3://ocrstorage4d/
├── run_20251129_abc123/
│   ├── detection_001/
│   │   ├── crop.jpg (original crop)
│   │   ├── crop_border.jpg (padded for OCR)
│   │   └── metadata.json
│   ├── detection_002/
│   │   └── crop.jpg
│   └── ... (up to 9 per batch)
└── run_20251129_def456/
    └── ...
```

---

### AWS IoT → Lambda → n8n

**Trigger Chain**:
```
1. Message published to: pi-imaging/detections (MQTT)
   ↓
2. AWS IoT Rule matches SELECT *
   ↓
3. Lambda function triggered: pi_imaging_to_n8n
   - Receives: MQTT payload (JSON)
   - Parses: run_id, detection_id, s3_keys
   - Prepares: Webhook payload
   ↓
4. HTTP POST to n8n webhook
   URL: {N8N_WEBHOOK_URL}/webhook/simcard-ocr
   Body: {
     "run_id": "...",
     "batch_id": "...",
     "crops": ["s3://...", "s3://...", ...],
     "timestamp": "2025-11-29T10:30:45Z"
   }
   ↓
5. n8n Workflow Executes:
   - Download image from S3
   - Run Tesseract OCR
   - Extract ICCID, phone number
   - Validate format
   - Insert into Airtable
   - Log results
```

**Lambda Code Location**: `Unit_test_code/lambda_pi_imaging_to_n8n.py`

---

## Communication Protocols

### 1. WebSocket (Browser ↔ Flask)

**Connection**:
```
Browser connects: ws://raspberry-pi:5000/socket.io/
Socket.IO handshake
└─ Authenticated: Browser receives connection_confirmed
```

**Event Messages**:
```python
# FROM SERVER → BROWSER
socketio.emit('frame_update', {
    'frame': 'base64_jpeg_data',
    'resolution': '1280x720',
    'timestamp': time.time(),
    'frame_count': 123
})

socketio.emit('performance_update', {
    'performance': {
        'images_per_minute': 92.5,
        'avg_latency_ms': 2100,
        'detection_accuracy': 0.95
    },
    'health': {
        'camera': 'operational',
        'mqtt': 'connected',
        's3': 'reachable'
    }
})

# FROM BROWSER → SERVER
client.emit('request_system_info')
client.emit('request_performance_metrics')
```

---

### 2. REST API (Browser/Client ↔ Flask)

**Authentication**: None (local network assumed secure)

**Standard Responses**:
```json
Success (200):
{
  "message": "System started",
  "run_id": "uuid",
  "status": "running"
}

Error (400/500):
{
  "error": "Error message",
  "details": "Detailed explanation",
  "troubleshooting": ["Step 1", "Step 2"]
}
```

---

### 3. Serial (Raspberry Pi ↔ ESP32)

**Protocol**: UART at 115200 baud
**Flow Control**: None (blocking reads with timeout)
**Format**: Text commands ending with newline

```
Tx: "DISPENSE 1\n"
Rx: "OK\n"

Tx: "MOVE 980 1\n"
Rx: "OK\n"

Timeout: 5 seconds (raises SerialException)
```

---

### 4. MQTT (Raspberry Pi ↔ AWS IoT Core)

**Protocol**: MQTT 3.1.1 over TLS 1.2
**Port**: 8883
**Keep-Alive**: 30 seconds
**QoS**: 1 (at-least-once)

**Message Format**:
```json
Topic: pi-imaging/detections
Payload: {
  "run_id": "20251129_abc123",
  "frame_number": 42,
  "batch_id": "batch_20251129_102945",
  "timestamp": "2025-11-29T10:29:45.123Z",
  "detected_cards": 9,
  "confidence_threshold": 0.80,
  "confidences": [0.95, 0.93, 0.92, 0.91, 0.90, 0.89, 0.88, 0.87, 0.86],
  "s3_bucket": "ocrstorage4d",
  "crops": [
    {
      "detection_id": "det_001",
      "s3_key": "run_20251129_abc123/detection_001/crop.jpg",
      "confidence": 0.95,
      "bbox": {
        "x": 100,
        "y": 150,
        "width": 150,
        "height": 100
      }
    },
    ...
  ]
}
```

---

## Deployment Topology

### Single-Machine Deployment

```
┌─────────────────────────────────────────────┐
│     Raspberry Pi 4 (8GB RAM)                │
│                                             │
│  ┌─────────────────────────────────────┐   │
│  │  Flask + SocketIO (Port 5000)       │   │
│  │  - REST API endpoints               │   │
│  │  - WebSocket server                 │   │
│  │  - Real-time streaming              │   │
│  └─────────────────────────────────────┘   │
│          ↑                  ↓               │
│  ┌─────────────────────────────────────┐   │
│  │  camera_system.py                   │   │
│  │  - Picamera2 interface              │   │
│  │  - Edge Impulse inference           │   │
│  │  - S3 uploader (2 threads)          │   │
│  │  - MQTT publisher                   │   │
│  │  - Circuit breaker & retries        │   │
│  └─────────────────────────────────────┘   │
│          ↑ Serial (UART)                    │
│  ┌─────────────────────────────────────┐   │
│  │  Serial Device Handler              │   │
│  │  - ESP32 communication              │   │
│  │  - Command parsing                  │   │
│  │  - Response handling                │   │
│  └─────────────────────────────────────┘   │
│          ↓                                  │
└─────────────────────────────────────────────┘
         ↓                          ↓
    ┌─────────────┐          ┌─────────────┐
    │ ESP32 Board │          │ IR Camera   │
    │ + Motors    │          │ 2560×1440   │
    └─────────────┘          └─────────────┘
         ↓
    ┌─────────────────────────┐
    │ Conveyor + Dispensers   │
    │ - Stepper motor         │
    │ - 2 Servo motors        │
    │ - 9-card dispenser      │
    └─────────────────────────┘
```

### External Cloud Services

```
┌──────────────────┐
│  AWS IoT Core    │
│  (MQTT Broker)   │
│  eu-north-1      │
└─────────┬────────┘
          │ (MQTT Topic: pi-imaging/detections)
          ↓
┌──────────────────────┐
│  AWS Lambda          │
│  pi_imaging_to_n8n   │
└─────────┬────────────┘
          │ (HTTP POST)
          ↓
┌──────────────────────┐
│  n8n Workflow        │
│  (Oracle Cloud)      │
│  - OCR (Tesseract)   │
│  - Data validation   │
│  - Airtable insert   │
└──────────────────────┘

┌──────────────────────┐
│  AWS S3              │
│  ocrstorage4d        │
│  (Crop storage)      │
└──────────────────────┘

┌──────────────────────┐
│  Airtable            │
│  (Results DB)        │
└──────────────────────┘

┌──────────────────────┐
│  Power BI            │
│  (Analytics)         │
└──────────────────────┘
```

---

## Summary

The PI Imaging system employs a **layered, distributed architecture**:

1. **Hardware Layer**: ESP32 provides motor control via serial protocol
2. **Edge Intelligence Layer**: Raspberry Pi runs ML inference locally (no cloud latency)
3. **Real-time Interface**: WebSocket-based live dashboarding
4. **Cloud Integration**: AWS IoT Core for reliable messaging
5. **Workflow Automation**: Lambda + n8n for OCR and data processing
6. **Storage**: S3 for images, Airtable for structured data

This design achieves:
- ✅ **Low Latency**: 372ms inference on edge device
- ✅ **High Throughput**: 90+ cards/minute
- ✅ **Reliability**: Circuit breakers, message queuing, retries
- ✅ **Scalability**: Decoupled components via message queues
- ✅ **Observability**: Real-time metrics, health checks, detailed logging
