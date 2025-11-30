# Component Integration Guide

This document explains how the main components communicate and examples for integrating or replacing parts.

## ESP32 ↔ Raspberry Pi (Serial)
- Protocol: Plain-text commands with newline terminator
- Baud: 115200
- Commands: `DISPENSE`, `MOVE`, `POSITION`, `PUSH`, `RESET`, `TEST`

Example: send a dispense command in Python
```python
import serial
s = serial.Serial('/dev/ttyUSB0', 115200, timeout=5)
s.write(b'DISPENSE 1\n')
resp = s.readline().decode().strip()
print('Response:', resp)
```

## Raspberry Pi ↔ AWS (MQTT)
- Use X.509 certs; topic: `pi-imaging/detections`.
- Publish JSON payload with detection metadata (see `docs/ARCHITECTURE.md`).

## AWS → Lambda → n8n
- Lambda receives IoT message and forwards to n8n webhook.
- n8n workflow handles S3 download, OCR, validation and storage.

## Flask ↔ Frontend
- REST control endpoints (`/api/start`, `/api/stop`, etc.)
- Socket.IO WebSocket events for real-time media and metrics

## Replacing the model
- If you bring your own model (BYOM), follow Edge Impulse BYOM guidelines.
- Ensure inference output matches expected format: list of `{bbox, confidence, class_id}`.

## Adding another consumer
- To add a second downstream consumer, publish to another MQTT topic or extend the Lambda to fan-out HTTP requests.

## Tips for integration testing
- Use `--simulate True` flag to run Pi processes without hardware.
- Mock S3 using `localstack` for local integration tests.

