#!/usr/bin/env python3
"""
ESP32 Full System Test - Stepper + Servo Integration
Tests complete automation sequence for SIM card dispenser
"""

import serial
import time
import sys

ESP32_PORT = '/dev/ttyUSB0'
BAUD_RATE = 115200

def connect_esp32():
    """Connect to ESP32"""
    try:
        ser = serial.Serial(ESP32_PORT, BAUD_RATE, timeout=2)
        time.sleep(2)

        if ser.in_waiting:
            msg = ser.readline().decode('utf-8', errors='ignore').strip()
            print(f"ESP32: {msg}")

        return ser
    except Exception as e:
        print(f"✗ Error connecting: {e}")
        return None

def send_command(ser, cmd, wait_time=1):
    """Send command and read response"""
    print(f"\n→ Sending: {cmd}")
    ser.write(f"{cmd}\n".encode())
    time.sleep(wait_time)

    response = ""
    while ser.in_waiting:
        response += ser.readline().decode('utf-8', errors='ignore').strip() + " "

    print(f"← Response: {response.strip()}")
    return "OK" in response

def test_stepper(ser):
    """Test stepper motor movement"""
    print("\n" + "="*60)
    print("TEST 1: STEPPER MOTOR (Conveyor Belt)")
    print("="*60)

    # Small movement test
    print("\nMoving belt 200 steps...")
    if send_command(ser, "MOVE 200 1", wait_time=2):
        print("✓ Stepper motor working")
        return True
    else:
        print("✗ Stepper motor failed")
        return False

def test_position_servo(ser):
    """Test platform positioning servo"""
    print("\n" + "="*60)
    print("TEST 2: POSITION SERVO (Platform Movement)")
    print("="*60)

    # Test position 0 (first row)
    print("\nMoving to Position 0 (First Row)...")
    if not send_command(ser, "POSITION 0", wait_time=1):
        print("✗ Position 0 failed")
        return False

    time.sleep(1)

    # Test position 90 (second row)
    print("\nMoving to Position 90 (Second Row)...")
    if not send_command(ser, "POSITION 90", wait_time=1):
        print("✗ Position 90 failed")
        return False

    time.sleep(1)

    # Return to home
    print("\nReturning to Position 0...")
    if send_command(ser, "POSITION 0", wait_time=1):
        print("✓ Position servo working")
        return True
    else:
        print("✗ Position servo failed")
        return False

def test_pusher_servo(ser):
    """Test SIM card pusher servo"""
    print("\n" + "="*60)
    print("TEST 3: PUSHER SERVO (Card Dispensing)")
    print("="*60)

    # Test retracted position
    print("\nRetracted (0°)...")
    if not send_command(ser, "PUSH 0", wait_time=1):
        print("✗ Retract failed")
        return False

    time.sleep(1)

    # Test extended position
    print("\nExtended (90°) - Pushing cards...")
    if not send_command(ser, "PUSH 90", wait_time=1):
        print("✗ Push failed")
        return False

    time.sleep(1)

    # Return to retracted
    print("\nRetracting...")
    if send_command(ser, "PUSH 0", wait_time=1):
        print("✓ Pusher servo working")
        return True
    else:
        print("✗ Pusher servo failed")
        return False

def test_full_sequence(ser):
    """Test complete automation sequence - 2 cycles of 6 cards each"""
    print("\n" + "="*60)
    print("TEST 4: FULL AUTOMATION SEQUENCE")
    print("="*60)

    print("\n=== FRAME 1: Drop 6 cards (2 rows) ===")

    # Row 1: Position 0, push, retract
    print("\n[1/4] Row 1: Position servo to 0°, push cards, retract")
    if not send_command(ser, "POSITION 0", wait_time=1):
        return False
    if not send_command(ser, "PUSH 90", wait_time=1):
        return False
    if not send_command(ser, "PUSH 0", wait_time=1):  # Retract for gravity drop
        return False

    # Row 2: Position 90, push, retract
    print("\n[2/4] Row 2: Position servo to 90°, push cards, retract")
    if not send_command(ser, "POSITION 90", wait_time=1):
        return False
    if not send_command(ser, "PUSH 90", wait_time=1):
        return False
    if not send_command(ser, "PUSH 0", wait_time=1):  # Retract for gravity drop
        return False

    print("\n→ 6 cards now on belt in Frame 1")

    # Move belt 10cm (6 cards go under camera, empty frame comes under dispenser)
    print("\n[3/4] Move belt 10cm (6 cards → camera, new frame → dispenser)")
    if not send_command(ser, "MOVE 980 1", wait_time=3):
        return False

    print("\n=== FRAME 2: Drop 6 cards (2 rows) ===")

    # Servo still at Row 2 position - push from there first
    print("\n[4/4] Row 2: (Already at 90°) push cards, retract")
    if not send_command(ser, "PUSH 90", wait_time=1):
        return False
    if not send_command(ser, "PUSH 0", wait_time=1):  # Retract for gravity drop
        return False

    # Row 1: Position 0, push, retract
    print("\n[5/4] Row 1: Position servo to 0°, push cards, retract")
    if not send_command(ser, "POSITION 0", wait_time=1):
        return False
    if not send_command(ser, "PUSH 90", wait_time=1):
        return False
    if not send_command(ser, "PUSH 0", wait_time=1):  # Retract for gravity drop
        return False

    print("\n→ 6 cards now on belt in Frame 2")

    # Move belt 10cm (6 cards go under camera)
    print("\n[6/4] Move belt 10cm (6 cards → camera)")
    if not send_command(ser, "MOVE 980 1", wait_time=3):
        return False

    print("\n--- Reset servos to home ---")
    if send_command(ser, "RESET", wait_time=1):
        print("✓ Full automation sequence complete")
        print("✓ Total: 12 cards dispensed across 2 frames")
        return True
    else:
        print("✗ Reset failed")
        return False

def run_all_tests(ser):
    """Run all tests in sequence"""
    results = {
        "Stepper Motor": False,
        "Position Servo": False,
        "Pusher Servo": False,
        "Full Sequence": False
    }

    # Run tests
    results["Stepper Motor"] = test_stepper(ser)
    time.sleep(2)

    results["Position Servo"] = test_position_servo(ser)
    time.sleep(2)

    results["Pusher Servo"] = test_pusher_servo(ser)
    time.sleep(2)

    results["Full Sequence"] = test_full_sequence(ser)

    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for test, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test:20s} : {status}")

    all_passed = all(results.values())
    print("\n" + "="*60)
    if all_passed:
        print("✓✓✓ ALL TESTS PASSED - SYSTEM READY ✓✓✓")
    else:
        print("✗✗✗ SOME TESTS FAILED - CHECK HARDWARE ✗✗✗")
    print("="*60)

    return all_passed

if __name__ == "__main__":
    print("ESP32 Full System Test - Stepper + Servos")
    print("="*60)

    esp32 = connect_esp32()
    if not esp32:
        print("\n✗ Cannot connect to ESP32")
        print("Check:")
        print("  - ESP32 plugged into Pi USB")
        print("  - Code uploaded to ESP32")
        print("  - Correct port: /dev/ttyUSB0")
        sys.exit(1)

    try:
        run_all_tests(esp32)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    finally:
        esp32.close()
        print("\nSerial connection closed")