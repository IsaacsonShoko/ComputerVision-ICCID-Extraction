// ESP32 Motor Controller + Servo Control for SIM Card Dispenser
// Optimized for Edge Impulse 10-item detection capability
// Dispenses 9 SIM cards (3 positions × 3 cards) to maximize ML model utilization
//
// Hardware Connections:
// - Stepper: GPIO 19 = STEP, GPIO 18 = DIR
// - Position Servo: GPIO 23 (controls platform angle: 0°, 90°, 180°)
// - Pusher Servo: GPIO 5 (pushes 3 SIM cards per position)
//
// Upload Instructions:
// 1. Install ESP32 board in Arduino IDE (File > Preferences > Additional Boards Manager URLs)
//    Add: https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
// 2. Tools > Board > ESP32 Dev Module
// 3. Tools > Port > Select your COM port
// 4. Upload this sketch

#define STEP 19
#define DIR 18
#define SERVO_POSITION 23  // Moves dispenser platform (3 positions: 0°, 90°, 180°)
#define SERVO_PUSHER 5     // Pushes 3 SIM cards simultaneously per position

// PWM channels for servos (ESP32 has 16 channels: 0-15)
// Using higher channels to avoid conflicts with GPIO 18/19
#define POSITION_CHANNEL 8
#define PUSHER_CHANNEL 9

// PWM properties
#define PWM_FREQ 50        // 50Hz for servos
#define PWM_RESOLUTION 16  // 16-bit resolution

void setup() {
  Serial.begin(115200);
    
  // Stepper pins - set first
  pinMode(STEP, OUTPUT);
  pinMode(DIR, OUTPUT);
  digitalWrite(STEP, LOW);
  digitalWrite(DIR, LOW);
    
  // Configure PWM channels for servos
  ledcSetup(POSITION_CHANNEL, PWM_FREQ, PWM_RESOLUTION);
  ledcSetup(PUSHER_CHANNEL, PWM_FREQ, PWM_RESOLUTION);
    
  // Attach pins to channels
  ledcAttachPin(SERVO_POSITION, POSITION_CHANNEL);
  ledcAttachPin(SERVO_PUSHER, PUSHER_CHANNEL);
    
  // Re-configure stepper pins after servo setup (just to be safe)
  pinMode(STEP, OUTPUT);
  pinMode(DIR, OUTPUT);
    
  // Initialize servos to retracted position (0 degrees)
  servoWrite(POSITION_CHANNEL, 0);
  servoWrite(PUSHER_CHANNEL, 0);
    
  Serial.println("READY - SIM Card Dispenser");
}

// Write angle to servo using LEDC
void servoWrite(int channel, int angle) {
  // Constrain angle to valid range
  angle = constrain(angle, 0, 180);
    
  // Calculate duty cycle for SG90 servo
  // 0° = 1ms pulse (2.5% duty) = ~1638 (at 16-bit, 50Hz)
  // 180° = 2ms pulse (12.5% duty) = ~8192 (at 16-bit, 50Hz)
  int dutyCycle = map(angle, 0, 180, 1638, 8192);
    
  ledcWrite(channel, dutyCycle);
}

void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
        
    // STEPPER MOTOR COMMAND: MOVE <steps> <direction>
    if (cmd.startsWith("MOVE")) {
      int s1 = cmd.indexOf(' ');
      int s2 = cmd.indexOf(' ', s1 + 1);
      int steps = cmd.substring(s1 + 1, s2).toInt();
      digitalWrite(DIR, cmd.substring(s2 + 1).toInt());
            
      for (int i = 0; i < steps; i++) {
        digitalWrite(STEP, HIGH);
        delayMicroseconds(1000);
        digitalWrite(STEP, LOW);
        delayMicroseconds(1000);
      }
      Serial.println("OK");
    }
        
    // POSITION SERVO: POSITION <angle>
    // Example: POSITION 90 (move platform to second position)
    else if (cmd.startsWith("POSITION")) {
      int angle = cmd.substring(9).toInt();
      angle = constrain(angle, 0, 180);
      servoWrite(POSITION_CHANNEL, angle);
      delay(500);  // Wait for servo to reach position
      Serial.println("OK");
    }
        
    // PUSHER SERVO: PUSH <angle>
    // Example: PUSH 35 (extend pusher to dispense cards - short distance)
    else if (cmd.startsWith("PUSH")) {
      int angle = cmd.substring(5).toInt();
      angle = constrain(angle, 0, 180);
      servoWrite(PUSHER_CHANNEL, angle);
      delay(400);  // Wait for servo to reach position
      Serial.println("OK");
    }
        
    // DISPENSE SEQUENCE: DISPENSE <position>
    // Automated sequence: move platform + push cards
    // DISPENSE 1 = first row (3 cards) at 0°
    // DISPENSE 2 = second row (3 cards) at 90°
    // DISPENSE 3 = third row (3 cards) at 180°
    // Total: 9 cards to maximize Edge Impulse 10-item detection
    else if (cmd.startsWith("DISPENSE")) {
      int pos = cmd.substring(9).toInt();

      if (pos == 1) {
        // First row: Position 0°, Push out
        Serial.println("Dispensing row 1 (0 deg)...");
        servoWrite(POSITION_CHANNEL, 0);
        delay(300);
        servoWrite(PUSHER_CHANNEL, 35);   // Push cards out (short distance)
        delay(400);
        servoWrite(PUSHER_CHANNEL, 0);    // Retract for gravity feed
        delay(300);
        Serial.println("OK");
      }
      else if (pos == 2) {
        // Second row: Position 90°, Push out
        Serial.println("Dispensing row 2 (90 deg)...");
        servoWrite(POSITION_CHANNEL, 90);
        delay(300);
        servoWrite(PUSHER_CHANNEL, 35);   // Push cards out (short distance)
        delay(400);
        servoWrite(PUSHER_CHANNEL, 0);    // Retract for gravity feed
        delay(300);
        Serial.println("OK");
      }
      else if (pos == 3) {
        // Third row: Position 180°, Push out
        Serial.println("Dispensing row 3 (180 deg)...");
        servoWrite(POSITION_CHANNEL, 180);
        delay(300);
        servoWrite(PUSHER_CHANNEL, 35);   // Push cards out (short distance)
        delay(400);
        servoWrite(PUSHER_CHANNEL, 0);    // Retract for gravity feed
        delay(300);
        Serial.println("OK");
      }
      else {
        Serial.println("ERROR: Invalid position (use 1, 2, or 3)");
      }
    }

    // DISPENSE_ALL: Dispense all 9 cards in sequence
    // Automated full cycle for maximum throughput
    else if (cmd == "DISPENSE_ALL") {
      Serial.println("Dispensing all 9 cards...");

      // Row 1: Position 0°
      Serial.println("Row 1 (0 deg)...");
      servoWrite(POSITION_CHANNEL, 0);
      delay(300);
      servoWrite(PUSHER_CHANNEL, 35);
      delay(400);
      servoWrite(PUSHER_CHANNEL, 0);
      delay(300);

      // Row 2: Position 90°
      Serial.println("Row 2 (90 deg)...");
      servoWrite(POSITION_CHANNEL, 90);
      delay(300);
      servoWrite(PUSHER_CHANNEL, 35);
      delay(400);
      servoWrite(PUSHER_CHANNEL, 0);
      delay(300);

      // Row 3: Position 180°
      Serial.println("Row 3 (180 deg)...");
      servoWrite(POSITION_CHANNEL, 180);
      delay(300);
      servoWrite(PUSHER_CHANNEL, 35);
      delay(400);
      servoWrite(PUSHER_CHANNEL, 0);
      delay(300);

      // Return to home
      servoWrite(POSITION_CHANNEL, 0);
      delay(300);

      Serial.println("OK - 9 cards dispensed");
    }
    
    // RESET: Return all to home position
    else if (cmd == "RESET") {
      Serial.println("Resetting to home position...");
      servoWrite(POSITION_CHANNEL, 0);
      servoWrite(PUSHER_CHANNEL, 0);
      delay(500);
      Serial.println("OK");
    }
    
    // TEST: Run a full test cycle (all 3 positions)
    else if (cmd == "TEST") {
      Serial.println("Running test sequence (3 positions for 9 cards)...");

      // Test position servo through all 3 positions
      Serial.println("Testing position servo (0, 90, 180 deg)...");
      servoWrite(POSITION_CHANNEL, 0);
      delay(800);
      servoWrite(POSITION_CHANNEL, 90);
      delay(800);
      servoWrite(POSITION_CHANNEL, 180);
      delay(800);
      servoWrite(POSITION_CHANNEL, 0);
      delay(800);

      // Test pusher servo
      Serial.println("Testing pusher servo...");
      servoWrite(PUSHER_CHANNEL, 35);
      delay(800);
      servoWrite(PUSHER_CHANNEL, 0);
      delay(800);

      // Test full dispense sequence (all 3 rows)
      Serial.println("Testing dispense row 1 (0 deg)...");
      servoWrite(POSITION_CHANNEL, 0);
      delay(300);
      servoWrite(PUSHER_CHANNEL, 35);
      delay(400);
      servoWrite(PUSHER_CHANNEL, 0);
      delay(500);

      Serial.println("Testing dispense row 2 (90 deg)...");
      servoWrite(POSITION_CHANNEL, 90);
      delay(300);
      servoWrite(PUSHER_CHANNEL, 35);
      delay(400);
      servoWrite(PUSHER_CHANNEL, 0);
      delay(500);

      Serial.println("Testing dispense row 3 (180 deg)...");
      servoWrite(POSITION_CHANNEL, 180);
      delay(300);
      servoWrite(PUSHER_CHANNEL, 35);
      delay(400);
      servoWrite(PUSHER_CHANNEL, 0);
      delay(500);

      // Return to home
      servoWrite(POSITION_CHANNEL, 0);
      Serial.println("Test complete - OK (9 card positions tested)");
    }
    
    else {
      Serial.println("ERROR: Unknown command");
      Serial.println("Commands:");
      Serial.println("  MOVE <steps> <dir>  - Move stepper motor");
      Serial.println("  POSITION <angle>    - Move platform servo (0-180)");
      Serial.println("  PUSH <angle>        - Move pusher servo (0-180)");
      Serial.println("  DISPENSE <1|2|3>    - Dispense row (3 cards each)");
      Serial.println("  DISPENSE_ALL        - Dispense all 9 cards");
      Serial.println("  RESET               - Return to home position");
      Serial.println("  TEST                - Run full test sequence");
    }
  }
}
