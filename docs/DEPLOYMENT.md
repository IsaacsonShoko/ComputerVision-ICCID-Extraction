# Deployment Guide

## Overview

This guide covers deploying PI Imaging to a Raspberry Pi 4 in a production environment. The system requires hardware setup, cloud configuration, and application deployment.

## Prerequisites

### Hardware Requirements

- Raspberry Pi 4 (8GB RAM recommended)
- ESP32 DevKit with microUSB cable
- IR Camera Module (2560×1440) with USB connection
- Stepper Motor + Motor Driver (HKD 4A)
- 2 Servo Motors (position + pusher)
- 12V Power Supply (5A minimum)
- Ethernet or WiFi connection
- MicroSD card (64GB+ recommended)

### Cloud Accounts

- AWS Account with permissions for:
  - S3 (bucket creation)
  - IoT Core (device certificates, rules)
  - Lambda (function creation)
- n8n workspace (self-hosted or cloud)
- Airtable account with API key
- Edge Impulse account (pre-trained model)

## Step 1: Raspberry Pi OS Setup

### 1.1 Flash Raspberry Pi OS

```bash
# On your computer, use Raspberry Pi Imager
# Download from: https://www.raspberrypi.com/software/
# Select:
# - OS: Raspberry Pi OS (64-bit recommended)
# - Storage: Your MicroSD card
# - Advanced options: Set hostname, enable SSH, set password
```

### 1.2 Initial Boot & Updates

```bash
# SSH into Raspberry Pi
ssh pi@raspberrypi.local

# Update system
sudo apt update
sudo apt upgrade -y

# Install system dependencies
sudo apt install -y python3-pip python3-venv git libatlas-base-dev
sudo apt install -y libjasper-dev libtiff5 libjasper1 libharfbuzz0b libwebp6
sudo apt install -y python3-dev libopenjp2-7 libtiff5 libjasper1 libharfbuzz0b

# Install OpenCV dependencies
sudo apt install -y libqtgui4 libqt4-test libhdf5-dev libharfbuzz0b libwebp6
```

### 1.3 Configure GPU Memory

```bash
sudo raspi-config
# Navigate to: Performance Options → GPU Memory
# Set to: 256MB (important for camera/ML)
# Save and reboot
sudo reboot
```

### 1.4 Enable Camera Interface

```bash
sudo raspi-config
# Navigate to: Interface Options → Camera
# Enable the camera interface
# Save and reboot
sudo reboot

# Verify camera is detected
vcgencmd get_camera
# Should output: supported=1 detected=1
```

## Step 2: Clone & Setup Application

### 2.1 Clone Repository

```bash
cd ~
git clone https://github.com/IsaacsonShoko/PI_IMAGING.git
cd PI_IMAGING
```

### 2.2 Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### 2.3 Install Python Dependencies

```bash
pip install -r requirements.txt

# This installs:
# - Flask
# - OpenCV (cv2)
# - Pillow (image processing)
# - pyserial (ESP32 communication)
# - boto3 (AWS SDK)
# - paho-mqtt (MQTT client)
# - websocket-client
# - python-dotenv (environment variables)
```

### 2.4 Install Edge Impulse Linux SDK

```bash
curl -fsSL https://cdn.edgeimpulse.com/firmware/linux/jetson.sh | bash
# This installs Edge Impulse runtime and dependencies

# Verify installation
edge-impulse-linux --version
```

### 2.5 Build Frontend (Optional for Dashboard)

```bash
cd frontend
npm install
npm run build
cd ..

# This creates optimized dist/ folder for web dashboard
```

## Step 3: AWS IoT Core Setup

### 3.1 Create IoT Policy

```bash
# In AWS Console: IoT Core → Policies → Create
# Policy name: pi-imaging-policy

# JSON Policy Document:
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "iot:*",
      "Resource": "*"
    }
  ]
}
```

### 3.2 Create IoT Thing & Certificates

```bash
# In AWS Console: IoT Core → Manage → Things → Create thing
# Thing name: raspberrypi-alpha

# Create certificate:
# IoT Core → Certificates → Create certificate
# - Choose "Create certificate" with auto-generated keys
# - Attach policy: pi-imaging-policy
# - Download certificate, private key, and root CA

# Save to: ~/PI_IMAGING/certs/
# - device.cert.pem (certificate)
# - device.private.key (private key)
# - AmazonRootCA1.pem (root CA)
```

### 3.3 Set Certificate Permissions

```bash
cd ~/PI_IMAGING/certs
chmod 400 device.private.key
chmod 444 device.cert.pem
chmod 444 AmazonRootCA1.pem
ls -la  # Verify permissions
```

### 3.4 Create S3 Bucket

```bash
# In AWS Console: S3 → Create bucket
# Bucket name: pi-imaging-crops-{random}
# Region: eu-north-1 (or your region)
# Block public access: ON
# Versioning: OFF
```

### 3.5 Create IoT Rule (MQTT → Lambda)

```bash
# In AWS Console: IoT Core → Message routing → Rules → Create rule
# Rule name: pi_imaging_to_n8n

# SQL statement:
SELECT * FROM 'pi-imaging/detections'

# Action: Lambda
# - Lambda function: Select your n8n bridge Lambda (or create new)
# - Role: Create or select role with Lambda invoke permission

# This rule triggers Lambda every time a message is published to pi-imaging/detections
```

## Step 4: Environment Configuration

### 4.1 Create .env File

```bash
cp .env.example .env
nano .env  # Or use your preferred editor
```

### 4.2 Populate .env Variables

```bash
# AWS S3
AWS_ACCESS_KEY_ID=your-access-key-here
AWS_SECRET_ACCESS_KEY=your-secret-key-here
AWS_REGION=eu-north-1
S3_BUCKET_NAME=pi-imaging-crops-{random}

# AWS IoT Core
AWS_IOT_ENDPOINT=your-endpoint.iot.eu-north-1.amazonaws.com
AWS_IOT_CERT=certs/device.cert.pem
AWS_IOT_KEY=certs/device.private.key
AWS_IOT_CA=certs/AmazonRootCA1.pem
AWS_IOT_CLIENT_ID=raspberrypi-alpha
AWS_IOT_TOPIC=pi-imaging/detections

# ESP32 Serial
SERIAL_PORT=/dev/ttyUSB0  # Check with: ls /dev/ttyUSB* /dev/ttyACM*
SERIAL_BAUD=115200

# Camera
CAMERA_RESOLUTION=2560x1440
CAMERA_FPS=30

# Flask
FLASK_ENV=production
FLASK_HOST=0.0.0.0
FLASK_PORT=5000

# n8n Webhook (optional, for Lambda bridge)
N8N_WEBHOOK_URL=https://your-n8n-instance.com/webhook/pi-imaging
```

## Step 5: Flash ESP32

### 5.1 Install Arduino CLI

```bash
cd ~
curl -fsSL https://raw.githubusercontent.com/arduino/arduino-cli/master/install.sh | sh
export PATH=$PATH:~/bin
```

### 5.2 Setup Arduino Environment

```bash
~/bin/arduino-cli config init
~/bin/arduino-cli core update-index
~/bin/arduino-cli core install esp32:esp32
```

### 5.3 Upload ESP32 Code

```bash
cd ~/PI_IMAGING/ESP32_Motor_Controller

# Compile
~/bin/arduino-cli compile --fqbn esp32:esp32:esp32 ESP32_Motor_Controller.ino

# Find serial port
ls /dev/ttyUSB* /dev/ttyACM*
# Note the port (e.g., /dev/ttyUSB0)

# Upload
~/bin/arduino-cli upload -p /dev/ttyUSB0 --fqbn esp32:esp32:esp32 ESP32_Motor_Controller.ino
```

### 5.4 Verify ESP32 Communication

```bash
# Test serial connection
python3 -c "
import serial
import time
s = serial.Serial('/dev/ttyUSB0', 115200, timeout=2)
time.sleep(2)
s.write(b'TEST\n')
time.sleep(1)
print(s.read_all().decode())
s.close()
"
# Should output "OK" and motor should move slightly
```

## Step 6: Deploy Edge Impulse Model

### 6.1 Download Model

```bash
# From Edge Impulse Studio:
# 1. Go to Deployment
# 2. Select "Raspberry Pi 4 / Linux"
# 3. Download as .zip

# Extract to models/
cd ~/PI_IMAGING
unzip ~/Downloads/ei-your-project-linux-armv7.zip -d models/
```

### 6.2 Test Inference

```bash
cd ~/PI_IMAGING
python3 -c "
from edge_impulse_linux.runner import ImpulseRunner
import cv2

runner = ImpulseRunner('models/your-model')
print('Model loaded successfully')
print(f'Input shape: {runner.input_shape}')
"
```

## Step 7: Create Systemd Service

### 7.1 Create Service File

```bash
sudo nano /etc/systemd/system/pi-imaging.service
```

### 7.2 Service Configuration

```ini
[Unit]
Description=PI Imaging - Edge AI SIM Card Detection
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/PI_IMAGING
Environment="PATH=/home/pi/PI_IMAGING/.venv/bin"
ExecStart=/home/pi/PI_IMAGING/.venv/bin/python app.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

### 7.3 Enable & Start Service

```bash
sudo systemctl daemon-reload
sudo systemctl enable pi-imaging
sudo systemctl start pi-imaging

# View logs
sudo journalctl -u pi-imaging -f

# Check status
sudo systemctl status pi-imaging
```

## Step 8: Configure n8n Webhook

### 8.1 Create n8n Workflow

In n8n:
1. Create new workflow
2. Add "Webhook" node as trigger
3. Configure:
   - Method: POST
   - Path: /pi-imaging
4. Get webhook URL (e.g., https://your-instance.com/webhook/pi-imaging)

### 8.2 Add n8n Nodes

```
Webhook (trigger)
    ↓
S3 Download (cropped images)
    ↓
OCR Processing (extract ICCID, phone)
    ↓
Data Validation
    ↓
Airtable Insert
    ↓
Error handling
```

### 8.3 Configure Lambda Bridge

Create Lambda function that receives IoT messages and forwards to n8n:

```python
import json
import urllib3
import os

http = urllib3.PoolManager()

def lambda_handler(event, context):
    # Extract detections from IoT message
    payload = {
        'batch_id': event.get('batch_id'),
        's3_bucket': os.environ['S3_BUCKET'],
        's3_prefix': f"crops/{event.get('batch_id')}/",
        'crop_urls': [d['s3_url'] for d in event.get('detections', [])],
        'timestamp': event.get('timestamp')
    }
    
    # Send to n8n
    n8n_url = os.environ['N8N_WEBHOOK_URL']
    r = http.request('POST', n8n_url, body=json.dumps(payload).encode('utf-8'))
    
    return {
        'statusCode': r.status,
        'body': json.dumps('Webhook triggered')
    }
```

## Step 9: Verify Deployment

### 9.1 Check All Components

```bash
# 1. Check ESP32 connection
python3 -c "
import serial
s = serial.Serial('/dev/ttyUSB0', 115200, timeout=2)
import time; time.sleep(2)
s.write(b'TEST\n')
time.sleep(1)
print('ESP32:', s.read_all().decode())
"

# 2. Check camera
python3 -c "
import cv2
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
print(f'Camera: {frame.shape if ret else \"FAILED\"}')"

# 3. Check Edge Impulse model
python3 -c "
from edge_impulse_linux.runner import ImpulseRunner
runner = ImpulseRunner('models/your-model')
print(f'Model loaded: OK')"

# 4. Check AWS credentials
aws s3 ls pi-imaging-crops-{random} --region eu-north-1

# 5. Check IoT MQTT
python3 -c "
import paho.mqtt.client as mqtt
import ssl

client = mqtt.Client('test')
client.tls_set('certs/AmazonRootCA1.pem', 'certs/device.cert.pem', 'certs/device.private.key', ssl.CERT_REQUIRED, ssl.PROTOCOL_TLSv1_2)
client.tls_insecure_set(False)
client.connect(os.getenv('AWS_IOT_ENDPOINT'), 8883, 60)
print('IoT connection: OK')
"
```

### 9.2 Test Full Pipeline

```bash
# 1. Start Flask server
source .venv/bin/activate
python app.py
# Should output: "Running on http://0.0.0.0:5000"

# 2. In another terminal, test API
curl http://localhost:5000/health
# Should return: {"status": "ok"}

# 3. Open dashboard
# Browser: http://raspberrypi.local:5000
# Should show live camera feed and stats

# 4. Manually trigger ESP32
python3 -c "
import serial
s = serial.Serial('/dev/ttyUSB0', 115200, timeout=2)
import time; time.sleep(2)
s.write(b'DISPENSE_ALL\n')
time.sleep(10)  # Wait for dispensing
print('Check dashboard for detections')
"
```

## Step 10: Production Hardening

### 10.1 Security

```bash
# 1. Change default password
passwd

# 2. Configure SSH keys
ssh-keygen -t ed25519
# Add public key to authorized_keys

# 3. Disable password SSH
sudo nano /etc/ssh/sshd_config
# Set: PasswordAuthentication no
sudo systemctl restart ssh

# 4. Enable firewall
sudo ufw enable
sudo ufw allow 22/tcp  # SSH
sudo ufw allow 5000/tcp  # Flask
sudo ufw status
```

### 10.2 Monitoring

```bash
# Install monitoring tools
sudo apt install -y htop iotop nethogs

# Monitor service
sudo journalctl -u pi-imaging -f --lines 50

# Monitor system
htop  # Press C to sort by CPU, M for memory
```

### 10.3 Backup Configuration

```bash
# Backup important files
cp .env .env.backup
cp certs/* certs-backup/
git commit -m "Deployment configuration"
```

## Troubleshooting

### Camera Not Detected

```bash
# Check if connected
libcamera-hello --list-cameras

# Check permissions
ls -la /dev/video*

# If needed, add user to groups
sudo usermod -a -G video pi
```

### ESP32 Upload Fails

```bash
# Check serial port
ls /dev/ttyUSB* /dev/ttyACM*

# Check ESP32 is in download mode
# (Most Arduino boards auto-reset, but some need manual button press)

# Try different board: esp32:esp32:esp32wroom32
```

### AWS Connection Issues

```bash
# Verify certificate files exist
ls -la certs/

# Check certificate validity
openssl x509 -in certs/device.cert.pem -text -noout

# Test MQTT connection with diagnostic tool
mosquitto_sub -h your-endpoint.iot.region.amazonaws.com \
  -p 8883 \
  -t "pi-imaging/detections" \
  --cert certs/device.cert.pem \
  --key certs/device.private.key \
  --cafile certs/AmazonRootCA1.pem \
  -d
```

### High CPU/Memory Usage

```bash
# Check running processes
ps aux | grep python

# Check memory leaks
free -h
top -o %MEM

# Profile inference latency
python3 -c "
import time
from edge_impulse_linux.runner import ImpulseRunner
runner = ImpulseRunner('models/your-model')
start = time.time()
# Run inference 10 times
for i in range(10):
    runner.classify([test_image])
print(f'Avg latency: {(time.time()-start)/10*1000}ms')
"
```

## Post-Deployment

### 10.4 Enable Auto-Updates

```bash
sudo apt install -y unattended-upgrades
sudo dpkg-reconfigure unattended-upgrades
```

### 10.5 Setup Log Rotation

```bash
sudo nano /etc/logrotate.d/pi-imaging
```

```
/home/pi/PI_IMAGING/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
}
```

### 10.6 Monitor Disk Space

```bash
# Check usage
df -h

# Clean old logs
sudo journalctl --vacuum=30d

# Setup disk space alert
sudo apt install -y mlocate
cron: 0 * * * * df -h | awk 'NR==2 {if($5 >= 80) print "Disk warning"}' | mail -s "Disk Alert" admin@example.com
```

## Verification Checklist

- [ ] Raspberry Pi OS installed and updated
- [ ] Python dependencies installed
- [ ] Edge Impulse model deployed
- [ ] ESP32 flashed and tested
- [ ] AWS S3 bucket created
- [ ] AWS IoT Core thing and certificates created
- [ ] IoT rule configured
- [ ] .env file populated correctly
- [ ] Flask server starts without errors
- [ ] Dashboard accessible at http://raspberrypi.local:5000
- [ ] ESP32 responds to serial commands
- [ ] Camera feed visible in dashboard
- [ ] S3 uploads working
- [ ] MQTT messages publishing to IoT Core
- [ ] n8n webhook receiving messages
- [ ] Airtable inserts successful
- [ ] Systemd service enabled and running
- [ ] Certificates have correct permissions
- [ ] Firewall configured
- [ ] Backups taken

Once all items are checked, your PI Imaging system is ready for production!
