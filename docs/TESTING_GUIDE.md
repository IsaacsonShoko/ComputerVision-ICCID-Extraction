# Testing Guide

This document lists unit, integration, and performance testing strategies for PI_IMAGING.

## Unit tests
- Location: `Unit_test_code/`
- Run a single test file:
```bash
python -m pytest Unit_test_code/EdgeAI_Detection_Test.py
```

## Integration tests
- Recommended tools: `localstack` (mock AWS), `pytest`, and `supertest` (for REST endpoints from Node.js)
- Test pipelines:
  - Camera capture → model inference (use sample frames)
  - Crop upload → S3 mock (localstack)
  - MQTT publish → Lambda invocation (simulate via local script)

## End-to-end tests
- Run full stack on a Pi in a controlled environment:
  - Start Flask: `python app.py`
  - Start ESP32 in test mode (or simulate serial)
  - Trigger `DISPENSE_ALL` and assert n8n receives webhook

## Performance tests
- Measure throughput for batches: capture N frames and measure total processed cards/min
- Tools:
  - `ab` / `wrk` for stress-testing REST API
  - Custom Python script to simulate high-frequency captures

## Test data
- Use a curated `tests/` folder with representative images (rotations, glare, overlaps)
- Ensure annotations align with images for evaluation metrics

## CI suggestions
- Run unit tests on each PR (GitHub Actions)
- Optional matrix: run tests on Python 3.9–3.11
- Run static checks: `flake8` and `mypy` (if types are used)

## Regression & validation
- Track model metrics (mAP, recall) in a `models/metrics.json` file after each retrain
- Add a lightweight validation script that runs the model on a fixed validation set and fails CI if mAP drops below a threshold

## Example performance script (simulate capture)
```python
import time
from camera_system import CameraSystem

cs = CameraSystem(simulate=True)
start = time.time()
count = 0
for i in range(100):
    cs.capture_and_process_optimized()
    count += 9
end = time.time()
print(f"Processed {count} cards in {end-start}s -> {count/((end-start)/60)} cards/min")
```

