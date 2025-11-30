# API Documentation

Complete REST API and WebSocket specification for PI Imaging system.

## Table of Contents
1. [REST API Endpoints](#rest-api-endpoints)
2. [WebSocket Events](#websocket-events)
3. [Request/Response Formats](#requestresponse-formats)
4. [Authentication & Security](#authentication--security)
5. [Error Handling](#error-handling)
6. [Performance Considerations](#performance-considerations)

---

## REST API Endpoints

### Base URL
```
http://[RASPBERRY_PI_IP]:5000
Example: http://192.168.1.15:5000
```

---

## System Control Endpoints

### POST /api/start
**Start the SIM card detection system**

**Request**:
```bash
curl -X POST http://192.168.1.15:5000/api/start \
  -H "Content-Type: application/json" \
  -d '{
    "service_provider": "MTN"
  }'
```

**Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `service_provider` | string | Yes | Carrier name (e.g., "MTN", "CellC", "Telkom", "Vodacom") |

**Response (200)**:
```json
{
  "message": "Enhanced OCR system started with MTN",
  "run_id": "20251129_a7f3c2b1",
  "service_provider": "MTN",
  "ocr_optimized": true,
  "status": "running"
}
```

**Errors**:
```json
// 400: Invalid input
{
  "error": "Service provider is required"
}

// 400: Already running
{
  "error": "System is already running"
}

// 400: Simulation mode
{
  "error": "Cannot start real system in simulation mode"
}

// 500: Hardware unavailable
{
  "error": "Camera system not available",
  "details": "Hardware initialization failed. Check camera and ESP32 connections.",
  "troubleshooting": [
    "Verify camera cable is connected",
    "Check ESP32 is connected via USB",
    "Ensure no other processes are using the camera",
    "Check system logs for detailed error messages"
  ]
}
```

---

### POST /api/stop
**Stop the detection system and return performance summary**

**Request**:
```bash
curl -X POST http://192.168.1.15:5000/api/stop
```

**Response (200)**:
```json
{
  "message": "Enhanced system stopped successfully",
  "final_image_count": 125,
  "performance_summary": {
    "total_frames_captured": 125,
    "total_detections": 1125,
    "images_per_minute": 95.5,
    "avg_latency_ms": 2150,
    "avg_upload_time": 450,
    "detection_accuracy": 0.956,
    "uptime_seconds": 79
  }
}
```

---

### POST /api/pause
**Pause the system without stopping it**

**Request**:
```bash
curl -X POST http://192.168.1.15:5000/api/pause
```

**Response (200)**:
```json
{
  "message": "System paused",
  "timestamp": "2025-11-29T10:30:45.123Z"
}
```

**Errors**:
```json
{
  "error": "System is not running"
}
```

---

### POST /api/resume
**Resume a paused system**

**Request**:
```bash
curl -X POST http://192.168.1.15:5000/api/resume
```

**Response (200)**:
```json
{
  "message": "System resumed",
  "timestamp": "2025-11-29T10:30:45.123Z"
}
```

---

### POST /api/capture-test
**Manually capture and process a test image**

**Request**:
```bash
curl -X POST http://192.168.1.15:5000/api/capture-test
```

**Response (200)**:
```json
{
  "message": "High-quality test image captured and processed",
  "run_id": "test_a7f3c2b1",
  "optimized_for_ocr": true
}
```

---

## Status & Monitoring Endpoints

### GET /api/status
**Get current system status and performance metrics**

**Request**:
```bash
curl http://192.168.1.15:5000/api/status
```

**Response (200)**:
```json
{
  "running": true,
  "paused": false,
  "current_run_id": "20251129_a7f3c2b1",
  "service_provider": "MTN",
  "image_count": 125,
  "total_runs": 15,
  "last_image_time": "2025-11-29T10:30:40.123Z",
  "error_message": null,
  "start_time": "2025-11-29T10:29:21.000Z",
  "performance": {
    "avg_capture_time": 33.2,
    "avg_upload_time": 450,
    "images_per_minute": 95.5,
    "stream_fps": 15.0
  }
}
```

---

### GET /api/performance
**Get detailed performance metrics**

**Request**:
```bash
curl http://192.168.1.15:5000/api/performance
```

**Response (200)**:
```json
{
  "timestamp": "2025-11-29T10:30:45.123Z",
  "system_metrics": {
    "frames_processed": 125,
    "cards_detected": 1125,
    "detections_per_frame": 9.0,
    "avg_confidence": 0.956,
    "detection_accuracy": 0.956,
    "false_positive_rate": 0.002,
    "throughput_cards_per_minute": 95.5,
    "avg_latency_ms": 2150,
    "p95_latency_ms": 2850,
    "p99_latency_ms": 3200,
    "s3_upload_success_rate": 0.998,
    "mqtt_publish_success_rate": 1.0
  },
  "streaming_metrics": {
    "connected_clients": 2,
    "streaming_active": true,
    "stream_resolution": "1280x720",
    "stream_quality": 95,
    "average_fps": 14.8
  }
}
```

---

### GET /api/health
**Health check with system diagnostics**

**Request**:
```bash
curl http://192.168.1.15:5000/api/health
```

**Response (200)**:
```json
{
  "status": "healthy",
  "timestamp": "2025-11-29T10:30:45.123Z",
  "simulation_mode": false,
  "system_running": true,
  "camera_available": true,
  "camera_status": "operational",
  "streaming_active": true,
  "connected_clients": 2,
  "performance_metrics": {
    "images_per_minute": 95.5,
    "detection_accuracy": 0.956,
    "uptime_seconds": 84
  },
  "node_red": {
    "url": "http://localhost:1880/webhook/screenshot",
    "reachable": true,
    "status_code": 200,
    "latency_ms": 15
  }
}
```

---

### GET /api/stats
**Get system statistics**

**Request**:
```bash
curl http://192.168.1.15:5000/api/stats
```

**Response (200)**:
```json
{
  "total_runs": 15,
  "current_run_images": 125,
  "current_service_provider": "MTN",
  "uptime": "01:23:45",
  "last_activity": "2025-11-29T10:30:40.123Z",
  "simulation_mode": false,
  "performance": {
    "avg_capture_time": 33.2,
    "avg_upload_time": 450,
    "images_per_minute": 95.5,
    "stream_fps": 15.0
  },
  "streaming": {
    "connected_clients": 2,
    "active": true,
    "quality": 95
  }
}
```

---

### GET /api/settings
**Get current system settings**

**Request**:
```bash
curl http://192.168.1.15:5000/api/settings
```

**Response (200)**:
```json
{
  "aws_region": "eu-north-1",
  "bucket_name": "ocrstorage4d",
  "webhook_url": "http://localhost:1880/webhook/screenshot",
  "node_red_url": "http://localhost:1880/webhook/screenshot",
  "simulation_mode": false,
  "ocr_optimized": true,
  "capture_resolution": "2560x1440",
  "capture_quality": 95,
  "stream_resolution": "1280x720",
  "stream_quality": 95
}
```

---

### GET /api/camera/feed
**Get high-quality MJPEG video stream (HTTP)**

**Request**:
```html
<img src="http://192.168.1.15:5000/api/camera/feed" />
```

**Response**: Multipart MJPEG stream
- **Content-Type**: `multipart/x-mixed-replace; boundary=frame`
- **Frame Rate**: 25 FPS
- **Quality**: 95%
- **Resolution**: Original camera resolution (2560×1440)

---

## WebSocket Events

### Connection

#### Client connects
```javascript
const socket = io('http://192.168.1.15:5000');

socket.on('connection_confirmed', (data) => {
  console.log(data);
  // {
  //   "timestamp": "2025-11-29T10:30:45.123Z",
  //   "simulation_mode": false,
  //   "system_ready": true,
  //   "streaming_active": true,
  //   "connected_clients": 5
  // }
});
```

---

### Real-time Updates

#### frame_update
**High-frequency: Live camera frames (15 FPS)**

```javascript
socket.on('frame_update', (data) => {
  console.log(data);
  // {
  //   "frame": "base64_encoded_jpeg_data",
  //   "resolution": "1280x720",
  //   "quality": 75,
  //   "timestamp": 1701244245.123,
  //   "frame_count": 1234
  // }
  
  // Render frame to canvas
  const image = new Image();
  image.src = 'data:image/jpeg;base64,' + data.frame;
  canvas.getContext('2d').drawImage(image, 0, 0);
});
```

---

#### performance_update
**Medium-frequency: Metrics (every 1 second)**

```javascript
socket.on('performance_update', (data) => {
  console.log(data);
  // {
  //   "performance": {
  //     "images_per_minute": 95.5,
  //     "avg_latency_ms": 2150,
  //     "detection_accuracy": 0.956,
  //     "s3_upload_success_rate": 0.998
  //   },
  //   "image_count": 125,
  //   "health": {
  //     "camera": "operational",
  //     "mqtt": "connected",
  //     "s3": "reachable"
  //   },
  //   "timestamp": "2025-11-29T10:30:45.123Z"
  // }
});
```

---

#### status
**Low-frequency: System state (on change)**

```javascript
socket.on('status', (data) => {
  console.log(data);
  // {
  //   "running": true,
  //   "image_count": 125,
  //   "service_provider": "MTN",
  //   "last_image_time": "2025-11-29T10:30:40.123Z"
  // }
});
```

---

### System Events

#### system_started
**Emitted when system starts**

```javascript
socket.on('system_started', (data) => {
  console.log(data);
  // {
  //   "service_provider": "MTN",
  //   "run_id": "20251129_a7f3c2b1",
  //   "timestamp": "2025-11-29T10:29:21.000Z"
  // }
});
```

---

#### system_stopped
**Emitted when system stops with final metrics**

```javascript
socket.on('system_stopped', (data) => {
  console.log(data);
  // {
  //   "final_image_count": 125,
  //   "performance_summary": {
  //     "total_frames_captured": 125,
  //     "total_detections": 1125,
  //     "images_per_minute": 95.5,
  //     "avg_latency_ms": 2150,
  //     "detection_accuracy": 0.956
  //   },
  //   "timestamp": "2025-11-29T10:30:45.123Z"
  // }
});
```

---

#### system_paused / system_resumed
**System control events**

```javascript
socket.on('system_paused', (data) => {
  console.log('System paused:', data.timestamp);
});

socket.on('system_resumed', (data) => {
  console.log('System resumed:', data.timestamp);
});
```

---

### Logging Events

#### log_update
**Real-time application logs**

```javascript
socket.on('log_update', (data) => {
  console.log(data);
  // {
  //   "message": "Frame 42: Detected 9 cards",
  //   "level": "info",
  //   "timestamp": "2025-11-29T10:30:40.123Z",
  //   "full_message": "[10:30:40] [INFO] Frame 42: Detected 9 cards",
  //   "performance": {
  //     "latency_ms": 2150,
  //     "confidence": 0.956
  //   },
  //   "system_status": {
  //     "running": true,
  //     "image_count": 125
  //   }
  // }
});
```

---

#### error_update
**Critical errors and alerts**

```javascript
socket.on('error_update', (data) => {
  console.error(data);
  // {
  //   "error_message": "S3 upload failed: Connection timeout",
  //   "level": "error",
  //   "timestamp": "2025-11-29T10:30:40.123Z",
  //   "system_context": {
  //     "operation": "s3_upload",
  //     "image_count": 125,
  //     "retry_count": 3
  //   }
  // }
});
```

---

#### exception_details
**Detailed exception information (for debugging)**

```javascript
socket.on('exception_details', (data) => {
  console.error(data);
  // {
  //   "error_details": {
  //     "operation": "s3_upload",
  //     "exception_type": "ClientError",
  //     "exception_message": "An error occurred (NoSuchBucket) when calling PutObject operation"
  //   },
  //   "timestamp": "2025-11-29T10:30:40.123Z"
  // }
});
```

---

### Client Request Events

#### request_system_info
**Request system information**

```javascript
socket.emit('request_system_info');

socket.on('system_info_response', (data) => {
  console.log(data);
  // {
  //   "simulation_mode": false,
  //   "camera_available": true,
  //   "system_status": { ... },
  //   "performance_metrics": { ... },
  //   "timestamp": "2025-11-29T10:30:45.123Z"
  // }
});
```

---

#### request_performance_metrics
**Request current performance data**

```javascript
socket.emit('request_performance_metrics');

socket.on('performance_metrics', (data) => {
  console.log(data);
  // {
  //   "current_metrics": {
  //     "images_per_minute": 95.5,
  //     "detection_accuracy": 0.956
  //   },
  //   "system_status": {
  //     "running": true,
  //     "image_count": 125,
  //     "throughput_target": "25+ images/minute"
  //   },
  //   "timestamp": "2025-11-29T10:30:45.123Z"
  // }
});
```

---

#### request_queue_status
**Request upload queue status**

```javascript
socket.emit('request_queue_status');

socket.on('queue_status', (data) => {
  console.log(data);
  // {
  //   "upload_queue_size": 2,
  //   "webhook_queue_size": 0,
  //   "max_queue_size": 5,
  //   "queue_full_events": 0,
  //   "dropped_frames": 0,
  //   "timestamp": "2025-11-29T10:30:45.123Z"
  // }
});
```

---

#### request_log_history
**Request recent log history**

```javascript
socket.emit('request_log_history');

socket.on('log_history', (data) => {
  console.log(data);
  // {
  //   "logs": [ /* array of previous logs */ ],
  //   "message": "Log history feature ready",
  //   "timestamp": "2025-11-29T10:30:45.123Z"
  // }
});
```

---

#### clear_error_alerts
**Clear all error alerts**

```javascript
socket.emit('clear_error_alerts');

socket.on('errors_cleared', (data) => {
  console.log('Errors cleared:', data.timestamp);
});
```

---

## Request/Response Formats

### Content Types

**All requests and responses use**:
```
Content-Type: application/json
```

---

### Standard Response Envelope

#### Success
```json
{
  "message": "Operation successful",
  "data": { /* operation-specific data */ },
  "timestamp": "2025-11-29T10:30:45.123Z",
  "request_id": "abc-123-def"
}
```

#### Error
```json
{
  "error": "Error description",
  "details": "Detailed explanation of what went wrong",
  "code": "ERROR_CODE",
  "troubleshooting": [
    "Step 1 to fix",
    "Step 2 to fix"
  ],
  "timestamp": "2025-11-29T10:30:45.123Z"
}
```

---

### Timestamp Format

All timestamps use ISO 8601 format:
```
2025-11-29T10:30:45.123Z
```

---

## Authentication & Security

### Current Implementation
- ✅ **Local network only**: Assumes secure LAN
- ✅ **HTTPS ready**: Can add with reverse proxy (nginx)
- ✅ **AWS credentials**: Use IAM roles or environment variables

### Production Recommendations

1. **Enable HTTPS**:
   ```bash
   # Use nginx reverse proxy
   # Or: gunicorn with SSL
   gunicorn --certfile=cert.pem --keyfile=key.pem app:app
   ```

2. **Add Authentication**:
   ```python
   from flask_httpauth import HTTPBasicAuth
   
   auth = HTTPBasicAuth()
   
   @app.route('/api/status')
   @auth.login_required
   def get_status():
       return jsonify(system_state)
   ```

3. **Firewall Rules**:
   ```bash
   sudo ufw allow 5000/tcp  # Flask server
   sudo ufw allow 22/tcp    # SSH only
   sudo ufw default deny incoming
   ```

---

## Error Handling

### Common HTTP Status Codes

| Code | Meaning | Example |
|------|---------|---------|
| 200 | OK | Successful request |
| 400 | Bad Request | Missing required parameter |
| 404 | Not Found | Invalid endpoint |
| 500 | Internal Server Error | Camera/hardware failure |

---

### Error Response Example

```json
{
  "error": "Camera system not available",
  "details": "Hardware initialization failed. Check camera and ESP32 connections.",
  "code": "CAMERA_UNAVAILABLE",
  "troubleshooting": [
    "Verify camera cable is firmly connected",
    "Check ESP32 USB connection",
    "Restart Raspberry Pi",
    "Check system logs: sudo journalctl -u pi-imaging -f"
  ],
  "timestamp": "2025-11-29T10:30:45.123Z"
}
```

---

## Performance Considerations

### Rate Limiting
- **Frame updates**: 15 FPS (67ms between frames)
- **Performance updates**: 1 Hz (1 second)
- **Status queries**: Unlimited (cached in memory)

### Payload Sizes
```
frame_update (MJPEG 75%): ~40-60 KB per frame
performance_update: ~500 bytes
status: ~1 KB
```

### Bandwidth Requirements
```
Video stream (15 FPS @ 75%): ~7.5 Mbps
Logs + metrics: ~100 Kbps
Total typical usage: ~8 Mbps
```

---

## Example Client Implementation

### JavaScript/Node.js
```javascript
const io = require('socket.io-client');
const fetch = require('node-fetch');

const socket = io('http://192.168.1.15:5000');

// REST API call
async function startSystem(provider) {
  const response = await fetch('http://192.168.1.15:5000/api/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ service_provider: provider })
  });
  return response.json();
}

// WebSocket event
socket.on('frame_update', (frame) => {
  console.log(`Frame received: ${frame.frame_count}`);
});

// Interval polling
setInterval(async () => {
  const status = await fetch('http://192.168.1.15:5000/api/status')
    .then(r => r.json());
  console.log(`Processed: ${status.image_count} images`);
}, 5000);
```

---

## Summary

The API provides complete control and monitoring:
- ✅ **System Control**: Start, stop, pause, resume
- ✅ **Real-time Monitoring**: WebSocket events for live updates
- ✅ **Performance Metrics**: Detailed throughput and accuracy data
- ✅ **Health Checks**: Diagnostics for troubleshooting
- ✅ **Video Streaming**: MJPEG and low-latency JPEG over WebSocket
