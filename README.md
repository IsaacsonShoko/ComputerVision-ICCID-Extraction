# PI Imaging - Edge AI SIM Card Detection & OCR System

## In Loving Memory of My Mother

This project is dedicated to my mother, who inspired me to pursue technology and innovation. Though she is no longer with us physically, her spirit lives on in every line of code and every breakthrough achieved here.

*"Innovation is not just about technology—it's about making the world better for those we love."*

---

## Edge Impulse AI Contest Submission

### The Innovation Story: From 15 Cards/Minute to 90+ Cards/Minute

This project demonstrates how **unlocking the power of Edge Impulse's FOMO model (320x320)** transformed a simple conveyor-based SIM card scanner into a high-throughput automated detection and OCR system.

#### The Original Problem

The initial design was a **single-card-per-frame conveyor system**:
- Process 1 SIM card at a time
- Capture, detect, upload
- **Throughput: ~15 cards per minute**
- Bottleneck: Mechanical conveyor movement between captures

#### The Edge AI Breakthrough

When I deployed the **Edge Impulse FOMO (Faster Objects, More Objects) model**, I discovered it could detect **up to 10 objects per frame** with high accuracy. This completely changed the system architecture.

**Key Insight**: Why process 1 card per frame when the ML model can handle 10?

#### The New Architecture

Inspired by the FOMO model's multi-object detection capability, I designed a **9-card batch dispensing system**:

```
+--------------------------------------------------+
|           9-Card Batch Dispenser                 |
+--------------------------------------------------+
|                                                  |
|   Position 1 (0deg)    -> 3 cards dropped        |
|   Position 2 (90deg)   -> 3 cards dropped        |
|   Position 3 (180deg)  -> 3 cards dropped        |
|                                                  |
|   Total: 9 cards per frame capture               |
|                                                  |
+--------------------------------------------------+
```

**Why 9 cards?** The FOMO model supports 10 detections, but 9 provides a comfortable margin for overlapping cards and detection confidence.

#### Performance Improvement

| Metric | Original System | Edge AI System | Improvement |
|--------|-----------------|----------------|-------------|
| Cards per frame | 1 | 9 | **9x** |
| Throughput | 15 cards/min | 90+ cards/min | **6x** |
| Detection time | 100ms | 100ms | Same |
| Accuracy | 85% | 92% | +7% |

---

## System Architecture

### End-to-End Pipeline

```
+--------------+    +--------------+    +--------------+
|  ESP32       |    |  Raspberry   |    |  AWS IoT     |
|  Controller  |----|  Pi 4        |----|  Core        |
|              |    |              |    |              |
| - 2 Servos   |    | - Camera     |    | - MQTT       |
| - Stepper    |    | - FOMO Model |    | - Rules      |
| - Timing     |    | - Flask API  |    | - Lambda     |
+--------------+    +--------------+    +--------------+
                           |                    |
                           v                    v
                    +--------------+    +--------------+
                    |  AWS S3      |    |  n8n         |
                    |              |    |  (Oracle)    |
                    | - Cropped    |    |              |
                    |   SIM images |    | - OCR Flow   |
                    | - Metadata   |    | - Airtable   |
                    +--------------+    +--------------+
                                               |
                                               v
                                        +--------------+
                                        |  Power BI    |
                                        |  Dashboard   |
                                        |              |
                                        | - Analytics  |
                                        | - Reporting  |
                                        +--------------+
```

### Data Flow

1. **ESP32 Dispenser** -> Drops 9 SIM cards (3 positions x 3 cards)
2. **Raspberry Pi Camera** -> Captures high-res frame (2560x1440)
3. **FOMO Detection** -> Identifies all 9 cards with bounding boxes
4. **Individual Crops** -> Each detected card cropped and saved
5. **S3 Upload** -> Cropped images uploaded to AWS S3
6. **IoT MQTT** -> Metadata published to `pi-imaging/detections`
7. **Lambda Bridge** -> Forwards to n8n webhook
8. **n8n OCR Flow** -> Extracts text from SIM card images
9. **Airtable Storage** -> OCR results stored in database
10. **Power BI Dashboard** -> Real-time analytics visualization

---

## Key Components

### 1. Edge Impulse FOMO Model

**Model Specifications:**
- **Architecture**: FOMO (Faster Objects, More Objects)
- **Input Size**: 320x320 RGB
- **Objects per frame**: Up to 10
- **Inference time**: <100ms on Raspberry Pi 4
- **Training data**: 500+ labeled SIM card images

**Why FOMO over MobileNet SSD?**
- Faster inference for real-time processing
- Native multi-object detection
- Lower memory footprint
- Optimized for edge devices

### 2. ESP32 Motor Controller

**File**: `Unit_test_code/ESP32_Servo_Stepper_Motor_Timing.py`

Controls the 9-card dispensing mechanism:

| Component | GPIO | Function |
|-----------|------|----------|
| Stepper STEP | 19 | Conveyor movement |
| Stepper DIR | 18 | Direction control |
| Position Servo | 23 | Platform angle (0deg/90deg/180deg) |
| Pusher Servo | 5 | Push 3 cards per position |

**Serial Commands:**
```
DISPENSE 1       # Row 1: 3 cards at 0deg
DISPENSE 2       # Row 2: 3 cards at 90deg
DISPENSE 3       # Row 3: 3 cards at 180deg
DISPENSE_ALL     # All 9 cards in sequence
TEST             # Full test cycle
RESET            # Return to home position
```

### 3. AWS IoT Core Integration

**Simplified Messaging Architecture:**
- All messages go through AWS IoT Core MQTT
- No webhook fallbacks - single reliable path
- QoS 1 (at least once) delivery guarantee

**Configuration** (`.env`):
```bash
AWS_IOT_ENDPOINT="your-endpoint.iot.eu-north-1.amazonaws.com"
AWS_IOT_CERT="certs/device.cert.pem"
AWS_IOT_KEY="certs/device.private.key"
AWS_IOT_CA="certs/AmazonRootCA1.pem"
AWS_IOT_CLIENT_ID="raspberrypi-alpha"
AWS_IOT_TOPIC="pi-imaging/detections"
```

**IoT Rule**: `pi_imaging_to_n8n`
- SQL: `SELECT * FROM 'pi-imaging/detections'`
- Action: Triggers Lambda function
- Lambda forwards to n8n HTTP endpoint

### 4. n8n OCR Extraction Flow

**File**: `Unit_test_code/n8n_OCR_Extraction_Flow.json`

Automated workflow for OCR processing:

1. **Webhook Trigger** -> Receives metadata from Lambda
2. **S3 Image Fetch** -> Downloads cropped SIM card image
3. **OCR Processing** -> Extracts text (ICCID, phone number, etc.)
4. **Data Validation** -> Validates extracted data format
5. **Airtable Insert** -> Stores results in database
6. **Error Handling** -> Logs failures for retry

### 5. SIMCard Analytics Dashboard

**File**: `Unit_test_code/SIMCard_Analytics_Dashboard.pbix`

Power BI dashboard for real-time analytics:

**Dashboard Pages:**

1. **Overview**
   - Total SIM cards processed
   - Processing rate (cards/minute)
   - Detection accuracy trend
   - Service provider breakdown

2. **OCR Results**
   - Extraction success rate
   - ICCID validation status
   - Phone number formats detected
   - Error categorization

3. **Performance Metrics**
   - System uptime
   - Queue depths over time
   - Error rates by type
   - Processing time distribution

4. **Inventory Tracking**
   - SIM cards by provider (MTN, CellC, Telkom, Vodacom)
   - Batch processing history
   - Daily/weekly/monthly trends

**Data Sources:**
- Airtable (via API connector)
- AWS CloudWatch metrics
- n8n workflow logs

---

## Real-time Web Dashboard

### Live Monitoring Features

**Statistics Panel:**
- SIM Cards Processed (individual crops)
- Frames Captured (batch captures)
- Detection Count (cards detected per frame)
- Detection Confidence (average %)
- Dispensed SIM Cards (ESP32 batches)
- Dispense Speed (seconds per batch)

**System Health:**
- Circuit Breaker Status (CLOSED/OPEN/HALF-OPEN)
- Queue Status (Upload/IoT queues)
- IoT Connection Status
- Error Count
- Pending Retries
- Dropped Frames

**Live Feed:**
- Real-time camera stream with detection overlays
- Bounding boxes around detected SIM cards
- Confidence scores displayed

---

## Installation & Setup

### Prerequisites

```bash
# Raspberry Pi
sudo apt update && sudo apt upgrade
sudo apt install python3-pip python3-venv git

# Node.js (for frontend build)
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs
```

### Installation

```bash
# Clone repository
git clone https://github.com/IsaacsonShoko/PI_IMAGING.git
cd PI_IMAGING

# Backend setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Frontend build
cd frontend
npm install
npm run build
cd ..
```

### Configuration

Create `.env` file:

```bash
# AWS S3 Storage
AWS_ACCESS_KEY_ID="your-access-key"
AWS_SECRET_ACCESS_KEY="your-secret-key"
AWS_REGION="eu-north-1"
S3_BUCKET_NAME="your-bucket-name"

# AWS IoT Core (MQTT)
AWS_IOT_ENDPOINT="your-endpoint.iot.eu-north-1.amazonaws.com"
AWS_IOT_CERT="certs/device.cert.pem"
AWS_IOT_KEY="certs/device.private.key"
AWS_IOT_CA="certs/AmazonRootCA1.pem"
AWS_IOT_CLIENT_ID="raspberrypi-alpha"
AWS_IOT_TOPIC="pi-imaging/detections"
```

### Running the System

```bash
# Activate environment
source .venv/bin/activate

# Start system
python app.py

# Access dashboard
# http://localhost:5000 (local)
# http://raspberry-pi-ip:5000 (network)
```

---

## Project Files

### Core Application
- `app.py` - Flask application entry point
- `camera_system.py` - Main camera and detection system

### Frontend
- `frontend/` - React TypeScript dashboard
- `frontend/src/pages/Index.tsx` - Main dashboard page
- `frontend/src/hooks/useSystemSocket.ts` - WebSocket integration

### Unit Test Code
- `Unit_test_code/ESP32_Servo_Stepper_Motor_Timing.py` - Motor timing tests
- `Unit_test_code/EdgeAI_Detection_Test.py` - Detection model tests
- `Unit_test_code/lambda_pi_imaging_to_n8n.py` - Lambda function code
- `Unit_test_code/n8n_OCR_Extraction_Flow.json` - n8n workflow
- `Unit_test_code/SIMCard_Analytics_Dashboard.pbix` - Power BI dashboard

### Configuration
- `.env.example` - Environment template
- `requirements.txt` - Python dependencies

---

## Performance Results

### Detection Performance
- **Model Accuracy**: 92% confidence on SIM card detection
- **Inference Time**: <100ms per frame
- **Multi-object Detection**: 9 cards per frame reliably
- **False Positive Rate**: <2%

### System Throughput
- **Original**: 15 cards/minute (single card per frame)
- **Edge AI**: 90+ cards/minute (9 cards per frame)
- **Improvement**: **6x throughput increase**

### Reliability Metrics
- **Circuit Breaker**: Automatic failover protection
- **Message Delivery**: QoS 1 with persistent retry
- **Uptime**: 99.5% system availability

---

## Technology Stack

### Hardware
- Raspberry Pi 4 (8GB)
- ESP32 DevKit
- IR Camera Module (2560x1440)
- Stepper Motor + 2 Servo Motors

### Software
- **ML**: Edge Impulse FOMO
- **Backend**: Python, Flask, OpenCV, Socket.IO
- **Frontend**: React, TypeScript, Tailwind CSS
- **Cloud**: AWS S3, AWS IoT Core, Lambda
- **Automation**: n8n (Oracle Cloud)
- **Analytics**: Power BI, Airtable

---

## Edge Impulse Integration

### Model Training
1. Collected 500+ SIM card images
2. Labeled bounding boxes in Edge Impulse Studio
3. Trained FOMO model with transfer learning
4. Deployed to Raspberry Pi via Edge Impulse Linux SDK

### Key Advantages
- **Real-time inference** on edge device
- **No cloud dependency** for detection
- **Low latency** (<100ms)
- **Multi-object support** (up to 10 per frame)

---

## Future Roadmap

### Phase 1: Core System (Completed)
- Edge AI detection
- 9-card batch dispensing
- AWS IoT integration
- Real-time dashboard

### Phase 2: OCR Pipeline (In Progress)
- n8n OCR workflow
- ICCID extraction
- Phone number detection
- Airtable storage

### Phase 3: Analytics (In Progress)
- Power BI dashboard
- Performance metrics
- Inventory tracking
- Trend analysis

### Phase 4: Enterprise Features (Planned)
- Multi-user access
- API integrations
- Mobile application
- Advanced analytics

---

## Acknowledgments

- **Edge Impulse** - For powerful, accessible ML tools
- **AWS** - For IoT Core and cloud infrastructure
- **n8n** - For workflow automation
- **Raspberry Pi Foundation** - For edge computing platform
- **My Mother** - For inspiring this journey

---

## Contact

- **GitHub**: [IsaacsonShoko/PI_IMAGING](https://github.com/IsaacsonShoko/PI_IMAGING)
- **Project**: Edge Impulse AI Contest Submission

---

*Built with Edge AI to transform SIM card processing from manual to automated, achieving 6x throughput improvement through intelligent multi-object detection.*
