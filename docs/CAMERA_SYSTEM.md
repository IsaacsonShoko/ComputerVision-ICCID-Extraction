# Camera System

This document explains how `camera_system.py` works, configuration variables, and how to run or extend it.

## Purpose
- Capture high-resolution frames for detection and OCR
- Maintain a dual-stream pipeline (stream + inference)
- Interface with ESP32 for dispenser control
- Upload crops to S3 and publish detection metadata

## Key configuration
- `CAMERA_RESOLUTION` (default `2560x1440`)
- `INFERENCE_INPUT` (`320x320`) — resized for model
- `SERIAL_PORT` (e.g., `/dev/ttyUSB0`), `SERIAL_BAUD` (115200)
- `S3_BUCKET_NAME`, `AWS_IOT_ENDPOINT`, other env vars in `.env`

## Main functions
- `capture_and_process_optimized()` — main loop: capture frame → inference → crops → upload → mqtt
- `detect_simcards(image)` — runs Edge Impulse runner; returns list of detections: {bbox, confidence}
- `crop_card(image, bbox, padding=8)` — returns JPEG bytes for crop
- `enqueue_upload(crop_bytes, metadata)` — adds to upload queue
- `publish_detection(metadata)` — publishes JSON to MQTT topic

## Running locally
```bash
source .venv/bin/activate
python camera_system.py --simulate False
```

`--simulate True` will bypass hardware calls and use local sample images for development.

## Debugging tips
- To visualize detections locally, enable `DEBUG_OVERLAY=True` and view the MJPEG stream at `/api/camera/feed`.
- For step-through debugging use `pdb` inside `detect_simcards()` on a saved test frame.

## Extending model input
- Edge Impulse input is 320×320; keep inference path at this resolution for correct results.
- For larger detection ranges add an image pyramid or multi-scale detector and merge boxes.

## Test utility
- `Unit_test_code/EdgeAI_Detection_Test.py` demonstrates running the model on sample images and validating outputs.

