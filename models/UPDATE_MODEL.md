# Updating the Edge Impulse TinyML Model

This guide explains how to update the SIM card detection model on the Raspberry Pi.

## Prerequisites

- Edge Impulse account with your project
- SSH access to the Raspberry Pi
- Node.js installed on the Pi

## Method 1: Edge Impulse CLI (Recommended)

### First-time Setup

```bash
# Install Edge Impulse CLI
npm install -g edge-impulse-cli

# Login to your Edge Impulse account
edge-impulse-login
```

### Download Latest Model

```bash
# Navigate to project directory
cd ~/PI_IMAGING

# Download the latest model directly to the models folder
edge-impulse-linux-runner --download ~/PI_IMAGING/models/simcard_detection.eim
```

This will:
1. Prompt you to select your project (if multiple)
2. Automatically build and download the Linux (AARCH64) deployment
3. Save it as `simcard_detection.eim`

### Using API Key (No Prompts)

For automated deployments, use your project's API key:

```bash
edge-impulse-linux-runner --download ~/PI_IMAGING/models/simcard_detection.eim --api-key YOUR_PROJECT_API_KEY
```

Find your API key in Edge Impulse Studio: **Dashboard > Keys**

## Method 2: Manual Download from Edge Impulse Studio

1. Go to [Edge Impulse Studio](https://studio.edgeimpulse.com)
2. Open your project
3. Navigate to **Deployment**
4. Select **Linux (AARCH64)** for Raspberry Pi 4
5. Click **Build**
6. Download the `.eim` file
7. Transfer to Pi:
   ```bash
   scp simcard_detection.eim pi@your-pi-ip:~/PI_IMAGING/models/
   ```

## After Updating

### Restart the Application

```bash
# If running as a service
sudo systemctl restart pi-imaging

# Or if running manually
# Stop the current app.py process (Ctrl+C) and restart
cd ~/PI_IMAGING
source .venv/bin/activate
python app.py
```

### Verify Model Loaded

Check the console output for:
```
Loading detection model: simcard_detection.eim
Edge Impulse Linux SDK loaded successfully
```

## Model Information

- **Format**: `.eim` (Edge Impulse Model for Linux)
- **Architecture**: AARCH64 (ARM 64-bit for Raspberry Pi 4)
- **Location**: `~/PI_IMAGING/models/simcard_detection.eim`
- **Purpose**: SIM card detection using SSD (Single Shot Detector)
- **Max Detections**: 10 items per frame (optimized for 9-card dispensing)

## Troubleshooting

### "Model not found" Error

Ensure the model is in the correct location:
```bash
ls -la ~/PI_IMAGING/models/simcard_detection.eim
```

### "Permission denied" Error

Make the model executable:
```bash
chmod +x ~/PI_IMAGING/models/simcard_detection.eim
```

### Edge Impulse CLI Not Found

Ensure npm bin is in your PATH:
```bash
export PATH=$PATH:~/.npm-global/bin
# Or add to ~/.bashrc for persistence
```

### Wrong Architecture

If you get architecture errors, ensure you selected **Linux (AARCH64)** in Edge Impulse Studio, not:
- Linux (x86_64) - for desktop Linux
- C++ MCU - for microcontrollers
- TensorFlow Lite - raw model without runner

## Version History

Track model versions by checking Edge Impulse Studio:
- **Dashboard > Versions** shows all deployment builds
- Each build includes accuracy metrics and training parameters

## Retraining Tips

When retraining the model:

1. **Add diverse training data** - Different lighting, angles, SIM card positions
2. **Balance classes** - Ensure equal representation
3. **Test on device** - Use Edge Impulse's live classification before deploying
4. **Check performance** - Target >90% accuracy, <100ms inference time
