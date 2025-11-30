# ESP32 Integration Summary - Camera System

## ✅ COMPLETED: ESP32 Serial Control Integration

### Changes Made to `camera_system.py`

#### 1. **Added Serial Communication**
- Imported `serial` and `serial.tools.list_ports` modules
- Added ESP32 connection instance and thread lock in `__init__`
- Removed GPIO direct control (no longer needed)

#### 2. **New Methods Added**

##### `init_esp32_connection()`
- Automatically detects ESP32 on USB ports (tries `/dev/ttyUSB0`, `/dev/ttyACM0`, etc.)
- Establishes 115200 baud serial connection
- Waits for "READY" message from ESP32
- Handles connection errors gracefully

##### `send_esp32_command(command, wait_for_ok=True)`
- Thread-safe command sending with lock
- Sends commands to ESP32 via serial
- Optionally waits for "OK" response
- Returns True/False for success/failure
- Error handling for connection issues

##### `reset_servos()`
- Sends "RESET" command to ESP32
- Returns all servos to home position (0°)
- Used at startup and shutdown

##### `dispense_sim_cards()`
- **Complete automation sequence for 6-card batches**
- Sequence:
  1. Send "DISPENSE 1" → Row 1 position (0°) → push 35° → retract 0°
  2. Send "DISPENSE 2" → Row 2 position (90°) → push 35° → retract 0°
  3. Move belt 10cm forward (980 steps)
- Logs each step with addLogEntry()
- Returns True/False for success/failure

##### `manual_dispense_row(row_number)`
- Manual testing function
- Dispenses 3 cards from specific row (1 or 2)
- Useful for testing individual row dispensers

##### `move_conveyor_belt_optimized(distance_multiplier=1.0)`
- **UPDATED**: Now uses ESP32 instead of GPIO
- Sends "MOVE <steps> 1" command to ESP32
- Calculates steps: `980 * distance_multiplier`
- Direction 1 = forward
- Still includes belt stabilization pause (1 second)
- Error handling and logging

#### 3. **Cleanup Enhanced**
- Added ESP32 cleanup in `cleanup()` method
- Sends "RESET" command to servos before closing
- Closes serial connection properly
- Handles cleanup errors gracefully

#### 4. **Removed Methods**
- ❌ `init_gpio_devices()` - No longer needed (ESP32 handles GPIO)
- ❌ `init_stepper_motor()` - No longer needed (ESP32 handles motor init)

---

### ESP32 Firmware Updates (`ESP32_Motor_Controller.ino`)

#### Changed Pusher Servo Angle
- **OLD**: 90° push angle
- **NEW**: 35° push angle (short distance for close mount)
- Reason: Pusher mounted close to dispenser slots, only needs short travel
- Benefits:
  - Faster operation
  - Less servo strain
  - Gentler on cards
  - More reliable

#### Updated DISPENSE Commands
```cpp
// DISPENSE 1 (Row 1)
positionServo.write(0);      // Position to Row 1
delay(300);
pusherServo.write(35);       // Push 35° (short distance)
delay(400);
pusherServo.write(0);        // Retract for gravity feed
delay(300);

// DISPENSE 2 (Row 2)
positionServo.write(90);     // Position to Row 2
delay(300);
pusherServo.write(35);       // Push 35° (short distance)
delay(400);
pusherServo.write(0);        // Retract for gravity feed
delay(300);
```

---

### Test Suite Updates (`ESP32_Full_System_Test.py`)

#### Updated All Pusher Commands
- Changed from `PUSH 90` to `PUSH 35` throughout
- Updated test descriptions to indicate "short distance"
- Maintains same test sequence logic

---

## 🎯 How It Works Now

### System Flow (Automated Dispensing)
```
1. Pi calls: camera_system.dispense_sim_cards()
2. Pi → ESP32: "DISPENSE 1"
3. ESP32: Position 0° → Push 35° → Retract 0° → OK
4. Pi → ESP32: "DISPENSE 2"
5. ESP32: Position 90° → Push 35° → Retract 0° → OK
6. Pi → ESP32: "MOVE 980 1"
7. ESP32: 980 steps forward → OK
8. Result: 6 cards under camera, ready for capture
```

### Communication Protocol
- **Baud Rate**: 115200
- **Command Format**: `COMMAND <params>\n`
- **Response Format**: `OK\n`
- **Port**: Auto-detected (typically `/dev/ttyUSB0`)

### Available Commands from Pi
```python
# Stepper motor
send_esp32_command("MOVE 980 1")        # Move 980 steps forward

# Position servo
send_esp32_command("POSITION 0")        # Row 1
send_esp32_command("POSITION 90")       # Row 2

# Pusher servo
send_esp32_command("PUSH 35")           # Extend (push cards)
send_esp32_command("PUSH 0")            # Retract (gravity feed)

# Automated sequences
send_esp32_command("DISPENSE 1")        # Row 1 complete sequence
send_esp32_command("DISPENSE 2")        # Row 2 complete sequence

# Reset
send_esp32_command("RESET")             # All servos to home (0°)
```

---

## 📋 Next Steps

### 1. Upload Updated Firmware to ESP32
- Open `ESP32_Motor_Controller.ino` in Arduino IDE
- Upload to ESP32
- Verify "READY" message in Serial Monitor

### 2. Test on Raspberry Pi
```python
# In camera_system.py
system = OptimizedCameraConveyorSystem()

# Test servo reset
system.reset_servos()

# Test manual row dispense
system.manual_dispense_row(1)  # Row 1
system.manual_dispense_row(2)  # Row 2

# Test full automation
system.dispense_sim_cards()    # 6 cards
```

### 3. Integration with Capture Loop
- Still TODO: Modify `capture_and_process_optimized()` to call `dispense_sim_cards()` automatically
- Will trigger every N frames or on demand

### 4. Update Web Interface (app.py)
- Add buttons for manual dispense testing
- Add servo status display
- Add reset servo button

---

## 🔧 Adjustable Parameters

If 35° doesn't work perfectly, you can adjust:

### In ESP32 Firmware
```cpp
pusherServo.write(30);   // Try 30° for even shorter push
pusherServo.write(40);   // Try 40° for slightly longer push
pusherServo.write(45);   // Try 45° for longer push
```

### In camera_system.py
```python
# For testing different angles
send_esp32_command("PUSH 30")  # Shorter
send_esp32_command("PUSH 40")  # Longer
```

---

## ⚡ Benefits Achieved

1. **Pi CPU Offloaded**: No more GPIO direct control, ESP32 handles all timing
2. **Thread-Safe**: Serial commands protected with locks
3. **Error Handling**: Graceful failures with logging
4. **Automated 6-Card Batches**: Leverages Edge Impulse's 9-10 card capacity
5. **Shorter Pusher Travel**: 35° instead of 90° (faster, gentler)
6. **Clean Shutdown**: Servos reset to home on cleanup

---

## 🎉 Status
**Integration Complete!** Ready for testing on Raspberry Pi.

Upload the updated ESP32 firmware, then test the dispense sequence!
