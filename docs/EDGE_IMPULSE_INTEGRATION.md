# Edge Impulse Integration

This document explains how the MobileNetV2 SSD FPN-Lite model was trained, tuned, exported and integrated into PI_IMAGING.

## Dataset
- 78 high-quality images collected (backgrounds + multiple SIM cards per frame)
- Annotation: bounding boxes around SIM cards in Edge Impulse Studio
- Strategy: multiple cards per frame to improve tinyML generalization for multi-object detection

## Training
- Model: MobileNetV2 SSD FPN-Lite (320×320 input)
- Optimizer: Adam / default Edge Impulse settings
- Learning rate: 0.015
- Training cycles / epochs: 50 (main run) + experiments via EON Tuner

## EON Tuner results (summary)
- Best run (`rgb-ssd-e7a`) achieved 95% F1-score with latency ~372ms
- Typical runs showed latency in 372–434ms range on Raspberry Pi 4

## Validation & Test Metrics
- Validation mAP: 79.15% (mAP@IoU=50: 96.20%)
- Test mAP: 75.77% (mAP@IoU=50: 97.01%)
- Recall@max_detections=10 ≈ 78–82%
- F1-score ≈ 95%

## Export & Deployment
1. In Edge Impulse Studio → Deployment → Linux (Raspberry Pi)
2. Download the Linux deployment package (zip)
3. Extract into `models/` and test with `edge-impulse-linux` runner

### Quick test
```bash
python3 -c "from edge_impulse_linux.runner import ImpulseRunner; r=ImpulseRunner('models/your-model'); print('Loaded')"
```

## Performance tuning
- Keep inference input at 320×320
- Use EON Tuner to sweep learning rate, epochs and preprocess steps
- Reduce JS/stream overlay resolution to minimize CPU overhead

## Tips for improving accuracy
- Collect more frames showing: overlapping cards, rotated cards, varied lighting
- Use synthetic augmentation to simulate glare and motion blur
- Add medium-scale crops to boost medium-area mAP

## Notes
- Model is float32 (no quantization) — offers better accuracy but higher latency.
- For production, consider int8 quantization or a Coral TPU for lower latency.

