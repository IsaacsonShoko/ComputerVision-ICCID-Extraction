import serial
import time

# ESP32 Serial connection
ESP32_PORT = '/dev/ttyUSB0'  # Adjust if needed
BAUD_RATE = 115200

# Hard-coded steps for 10cm (matches camera_system.py)
STEPS_FOR_10CM = 980

def connect_esp32():
    """Connect to ESP32 via serial"""
    try:
        ser = serial.Serial(ESP32_PORT, BAUD_RATE, timeout=2)
        time.sleep(2)  # Wait for ESP32 to boot

        # Read startup message
        if ser.in_waiting:
            msg = ser.readline().decode('utf-8', errors='ignore').strip()
            print(f"ESP32: {msg}")

        return ser
    except Exception as e:
        print(f"Error connecting to ESP32: {e}")
        print("Make sure ESP32 is plugged into Pi USB port")
        return None

def move_motor(ser, steps, direction=1):
    """Send move command to ESP32

    Args:
        ser: Serial connection
        steps: Number of steps to move
        direction: 1 = forward, 0 = backward
    """
    cmd = f"MOVE {steps} {direction}\n"
    print(f"Sending command: {cmd.strip()}")

    ser.write(cmd.encode())
    time.sleep(steps * 0.002 + 0.5)  # Wait for movement to complete

    # Read response
    if ser.in_waiting:
        response = ser.readline().decode('utf-8', errors='ignore').strip()
        print(f"ESP32 response: {response}")
        return response == "OK"

    return False

def move_10cm(ser):
    """Move exactly 10cm forward"""
    print(f"Moving 10cm forward ({STEPS_FOR_10CM} steps)...")
    success = move_motor(ser, STEPS_FOR_10CM, direction=1)

    if success:
        print("✓ Movement complete")
    else:
        print("✗ Movement failed")

    return success

if __name__ == "__main__":
    print("ESP32 Motor Test - 10cm Movement")
    print("=" * 50)

    # Connect to ESP32
    esp32 = connect_esp32()

    if not esp32:
        print("\nFailed to connect to ESP32!")
        print("Check:")
        print("  1. ESP32 plugged into Pi USB")
        print("  2. Code uploaded to ESP32")
        print("  3. Run: ls /dev/ttyUSB* /dev/ttyACM* to find port")
        exit(1)

    try:
        # Test movement
        move_10cm(esp32)

        print("\nTest complete!")

    except KeyboardInterrupt:
        print("\n\nProgram stopped by user")

    finally:
        esp32.close()
        print("Serial connection closed")