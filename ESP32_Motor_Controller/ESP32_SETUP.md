# ESP32 Motor Controller Setup

## Hardware Wiring

Connect ESP32 to HKD 4A stepper driver:

| ESP32 Pin | HKD Terminal | Purpose |
|-----------|--------------|---------|
| GPIO 19 | PUL+ | Step pulses |
| GPIO 18 | DIR+ | Direction |
| GPIO 23 | ENA+ | Enable |
| GND | PUL-, DIR-, ENA- | Common ground (connect to all three) |

Keep existing connections:
- 12V PSU → HKD VCC/GND
- Motor wires → HKD A+/A-/B+/B-

## Software Installation (on Raspberry Pi)

### 1. Install Arduino CLI

```bash
curl -fsSL https://raw.githubusercontent.com/arduino/arduino-cli/master/install.sh | sh
export PATH=$PATH:~/bin
```

### 2. Install ESP32 Board Support

```bash
~/bin/arduino-cli config init
~/bin/arduino-cli core update-index
~/bin/arduino-cli core install esp32:esp32
```

### 3. Upload Code to ESP32

```bash
cd ~/PI_IMAGING/ESP32_Motor_Controller
~/bin/arduino-cli compile --fqbn esp32:esp32:esp32 ESP32_Motor_Controller.ino
~/bin/arduino-cli upload -p /dev/ttyUSB0 --fqbn esp32:esp32:esp32 ESP32_Motor_Controller.ino
```

If `/dev/ttyUSB0` doesn't work, check available ports:
```bash
ls /dev/ttyUSB* /dev/ttyACM*
```

## Testing

### Test ESP32 communication:

```bash
python3 -c "import serial; s=serial.Serial('/dev/ttyUSB0',115200,timeout=2); import time; time.sleep(2); s.write(b'MOVE 100 1\n'); time.sleep(1); print(s.read_all().decode())"
```

You should see "OK" response and motor should move.

## Integration with camera_system.py

The ESP32 listens for serial commands at 115200 baud:

**Stepper Motor:**
- `MOVE <steps> <direction>` - Move motor (e.g., "MOVE 980 1" = 10cm forward)
- Direction: 1 = forward, 0 = backward
- Steps: 980 = 10cm (based on existing STEPS_FOR_10CM)

**Servo Dispensing (9 cards - 3 positions × 3 cards):**
- `DISPENSE 1` - Dispense row 1 (3 cards at 0°)
- `DISPENSE 2` - Dispense row 2 (3 cards at 90°)
- `DISPENSE 3` - Dispense row 3 (3 cards at 180°)
- `DISPENSE_ALL` - Dispense all 9 cards in sequence
- `POSITION <angle>` - Move platform servo (0-180)
- `PUSH <angle>` - Move pusher servo (0-180)
- `RESET` - Return all servos to home position
- `TEST` - Run full test sequence

Speed matches existing system: 1ms delay between steps (STEPPER_DELAY = 0.001)

**Why 9 Cards?**
The Edge Impulse SSD model supports detecting up to 10 items. Dispensing 9 cards maximizes ML model utilization, increasing throughput from 15 cards/min (manual) to 45+ cards/min (automated).
