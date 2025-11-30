# Troubleshooting Guide

This page aggregates common issues and fixes encountered while running PI_IMAGING.

## Quick checklist
- Ensure `.env` exists and is populated.
- Ensure `certs/` contains `device.cert.pem`, `device.private.key`, `AmazonRootCA1.pem`.
- Check `systemctl status pi-imaging` and `journalctl -u pi-imaging -f` for logs.
- Confirm serial port for ESP32: `ls /dev/ttyUSB* /dev/ttyACM*`.

## Camera issues
- Symptom: Camera not detected or `libcamera` errors.
  - Run: `vcgencmd get_camera` — expects `supported=1 detected=1`.
  - Ensure camera interface is enabled: `sudo raspi-config` → Interface Options → Camera.
  - Add user to `video` group: `sudo usermod -a -G video pi` and reboot.
  - Test capture: `libcamera-jpeg -o test.jpg`.

## ESP32 / Serial issues
- Symptom: No response from ESP32, dispenser doesn't move.
  - Check cables and power to stepper/servos.
  - Verify serial port: `ls /dev/ttyUSB*`.
  - Test serial manually (Linux):
    ```bash
    echo -e "TEST\n" > /dev/ttyUSB0
    cat /dev/ttyUSB0
    ```
  - If upload fails, ensure board in flash mode and correct `fqbn` for Arduino CLI.

## Model / Inference issues
- Symptom: No detections or low confidence.
  - Confirm model is present in `models/` and path in config.
  - Validate input shape: Edge Impulse runner input shape must match `320x320`.
  - Run a single inference test script to measure latency and outputs.
  - Check camera exposure/lighting—OCR is sensitive to glare and small crops.

## S3 Upload / AWS errors
- Symptom: S3 upload failures or `NoSuchBucket`.
  - Verify `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` are set in `.env`.
  - Check bucket name and region.
  - Use `aws s3 ls s3://<bucket>` to verify access.

## MQTT / AWS IoT issues
- Symptom: Messages not appearing in IoT Core.
  - Verify certificate files exist and have correct permissions.
  - Ensure `AWS_IOT_ENDPOINT` is set and pingable.
  - Use `mosquitto_pub` / `mosquitto_sub` for local tests with cert/key.
  - Check IoT rule and Lambda logs in CloudWatch.

## n8n / OCR pipeline issues
- Symptom: n8n not receiving webhook or OCR failing.
  - Confirm Lambda forwarded to correct n8n webhook URL.
  - In n8n, review webhook executions and input data.
  - Check OCR node (Tesseract) language/data files are installed.

## Performance / High CPU
- Symptom: Pi overloaded, inference slow.
  - Reduce stream resolution for WebSocket view (keep inference at 320x320).
  - Limit number of concurrent uploads.
  - Ensure GPU memory set to 256MB.
  - Consider switching to a Coral accelerator or a Raspberry Pi 5 for production.

## Logs & Diagnostics
- Key logs:
  - Flask & app logs: `journalctl -u pi-imaging -f`
  - Camera system logs: `logs/camera_system.log` (if configured)
  - System logs: `dmesg` / `sudo journalctl -k`

## When to restart services
- Restart Flask service after changing `.env` or code:
```bash
sudo systemctl restart pi-imaging
sudo journalctl -u pi-imaging -f
```

## Contact & Next Steps
If the issue persists, collect the following and open an issue:
- `journalctl -u pi-imaging --since "10 minutes ago"`
- Output of `ls -la certs/` and `.env` (redact secrets)
- A short description of the exact hardware and OS configuration

