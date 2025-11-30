from gpiozero import OutputDevice
from time import sleep

# Pin Definitions
DIR_PIN = OutputDevice(13)    # Direction pin
STEP_PIN = OutputDevice(19)   # Step pin
M0_PIN = OutputDevice(21)     # Micro-stepping pin 1
M1_PIN = OutputDevice(20)     # Micro-stepping pin 2
M2_PIN = OutputDevice(16)     # Micro-stepping pin 3

# Hard-coded steps for 10cm
STEPS_FOR_10CM = 980

def initialize_microstepping():
    """Initialize 1/4 microstepping mode"""
    M0_PIN.off()  # OFF
    M1_PIN.on()   # ON
    M2_PIN.off()  # OFF
    print("1/4 step mode initialized")

def move_10cm():
    """Move exactly 10cm forward"""
    print("Moving 10cm forward...")
    DIR_PIN.value = 0 # Forward direction

    for _ in range(STEPS_FOR_10CM):
        STEP_PIN.on()
        sleep(0.001)  # Faster delay
        STEP_PIN.off()
        sleep(0.001)  # Faster delay

    print("Movement complete")

try:
    initialize_microstepping()
    move_10cm()

except KeyboardInterrupt:
    print("\nProgram stopped by user")