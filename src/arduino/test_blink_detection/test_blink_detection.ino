/*
Basic testing file that listens on serial and blinks the onboard LED when a blink is detected.

Allows communication between Python + Arduino.

Usage:
  Python sends '1' over serial (9600 baud) when a blink is detected.
  Arduino blinks the onboard LED (pin 13) 3 times in response.
*/

const int LED_PIN = LED_BUILTIN;  // pin 13 on most boards
const int BLINK_COUNT = 3;
const int BLINK_ON_MS = 150;
const int BLINK_OFF_MS = 100;

void setup() {
    Serial.begin(9600);
    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, LOW);
}

void loop() {
    if (Serial.available() > 0) {
        char received = Serial.read();
        if (received == '1') {
            for (int i = 0; i < BLINK_COUNT; i++) {
                digitalWrite(LED_PIN, HIGH);
                delay(BLINK_ON_MS);
                digitalWrite(LED_PIN, LOW);
                delay(BLINK_OFF_MS);
            }
        }
    }
}
