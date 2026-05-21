// TRAINER KIT ESP

#include <FastLED.h>

// --- LED Configuration ---
#define LED_PIN     5       // Data pin for LED strip
#define NUM_LEDS    92      // Number of LEDs in your strip
#define CHIPSET     WS2811  // Or WS2812B, SK6812, etc.
#define COLOR_ORDER GRB     // Or RGB, BGR, etc.

#define BRIGHTNESS  200     // Default brightness (0-255)
#define FRAMES_PER_SECOND 60

// --- Fire2012 Parameters (if Fire mode is used) ---
#define COOLING     55  // How much does the fire cool down? Less cooling = hotter fire. (50-100)
#define SPARKING    120 // Chance (out of 255) that a new spark will be lit. Higher chance = more sparks. (50-200)

// Pin Diffuser Relay
#define DIFFUSER         2     

CRGB leds[NUM_LEDS];
bool gReverseDirection = false; // For Fire2012 or other directional animations

// --- State Management ---
enum LedMode {
  MODE_OFF,
  MODE_SOLID_COLOR,
  MODE_FIRE,
  MODE_HAPPY,
  MODE_SAD
  // Add more modes here if needed
};

LedMode currentMode = MODE_FIRE; // Default mode on startup
CRGB solidColor = CRGB::White;   // Default color for SOLID_COLOR mode
bool ledsAreOn = true;           // LEDs start ON by default

const int HUMIDIFIER_PIN = 12;      // Update to match the GPIO wired to the humidifier relay
const bool HUMIDIFIER_ACTIVE_LOW = true; // Set to false if the relay expects an active-HIGH signal
bool humidifierIsOn = false;

void writeHumidifierOutput() {
  const uint8_t level = humidifierIsOn
                          ? (HUMIDIFIER_ACTIVE_LOW ? LOW : HIGH)
                          : (HUMIDIFIER_ACTIVE_LOW ? HIGH : LOW);
  digitalWrite(HUMIDIFIER_PIN, level);
}

// void setHumidifier(bool turnOn) {
//   const bool stateChanged = humidifierIsOn != turnOn;
//   humidifierIsOn = turnOn;
//   writeHumidifierOutput();
//   Serial.print("Humidifier ");
//   if (stateChanged) {
//     Serial.print("turned ");
//   } else {
//     Serial.print("remains ");
//   }
//   Serial.println(humidifierIsOn ? "ON" : "OFF");
// }

void setHumidifier(bool turnOn) {
  const bool stateChanged = humidifierIsOn != turnOn;
  humidifierIsOn = turnOn;
  writeHumidifierOutput();
  Serial.print("Humidifier ");
  if (stateChanged) {
    Serial.print("Humidifier On ");
    digitalWrite(12, HIGH);
  } else {
    Serial.print("Humidifier Off ");
    digitalWrite(12, LOW);
  }
  Serial.println(humidifierIsOn ? "ON" : "OFF");
}

void setup() {
  delay(2000); // Sanity delay

  Serial.begin(115200); // Initialize serial communication
  Serial.println("------------------------------------");
  Serial.println("ESP32 LED Control Ready");
  Serial.println("Send commands via Serial (e.g., ON, OFF, MERAH, HAPPY)");
  Serial.println("Current Mode: FIRE");
  Serial.println("------------------------------------");

  FastLED.addLeds<CHIPSET, LED_PIN, COLOR_ORDER>(leds, NUM_LEDS).setCorrection(TypicalLEDStrip);
  FastLED.setBrightness(BRIGHTNESS);
  pinMode(HUMIDIFIER_PIN, OUTPUT);
  setHumidifier(false); // Ensure humidifier defaults to OFF at boot
}

void loop() {
  handleSerialCommands(); // Check for and process incoming serial commands

  if (ledsAreOn) {
    FastLED.setBrightness(BRIGHTNESS); // Ensure brightness is set if LEDs are on
    switch (currentMode) {
      case MODE_SOLID_COLOR:
        runSolidColorMode();
        break;
      case MODE_FIRE:
        runFireMode();
        break;
      case MODE_HAPPY:
        runHappyMode();
        break;
      case MODE_SAD:
        runSadMode();
        break;
      case MODE_OFF: // Should be handled by ledsAreOn, but as a fallback
        fill_solid(leds, NUM_LEDS, CRGB::Black);
        break;
    }
  } else { // ledsAreOn is false
    fill_solid(leds, NUM_LEDS, CRGB::Black);
    // Or you can use FastLED.setBrightness(0); and restore it in the ON command
  }

  FastLED.show();
  FastLED.delay(1000 / FRAMES_PER_SECOND);
}

void handleSerialCommands() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();          // Remove any leading/trailing whitespace
    command.toUpperCase();   // Convert to uppercase for case-insensitive comparison

    Serial.print("Received: ");
    Serial.println(command);

    bool commandRecognized = true;

    if (command == "ON") {
      ledsAreOn = true;
      currentMode = MODE_FIRE; // <<< MODIFIKASI DI SINI: Set mode ke FIRE saat ON
      Serial.println("LEDs turned ON. Mode set to FIRE.");
    } else if (command == "OFF") {
      ledsAreOn = false;
      currentMode = MODE_OFF; // Explicitly set mode to OFF
      Serial.println("LEDs turned OFF.");
    }
    // --- Solid Colors ---
    else if (command == "MERAH") {
      setSolidColor(CRGB::Red, "MERAH");
    } else if (command == "HIJAU") {
      setSolidColor(CRGB::Green, "HIJAU");
    } else if (command == "BIRU") {
      setSolidColor(CRGB::Blue, "BIRU");
    } else if (command == "LAVENDER") {
      setSolidColor(CRGB::Lavender, "LAVENDER");
    } else if (command == "MAGENTA") {
      setSolidColor(CRGB::Magenta, "MAGENTA");
    } else if (command == "PINK") {
      setSolidColor(CRGB::DeepPink, "PINK"); // Or CRGB::Pink
    } else if (command == "VIOLET") {
      setSolidColor(CRGB::Violet, "VIOLET"); // Or CRGB::Purple
    } else if (command == "AQUA") {
      setSolidColor(CRGB::Aqua, "AQUA"); // Or CRGB::Cyan
    } else if (command == "KUNING") {
      setSolidColor(CRGB::Yellow, "KUNING");
    } else if (command == "EMAS") {
      setSolidColor(CRGB::Gold, "EMAS");
    } else if (command == "ABU") {
      setSolidColor(CRGB::Gray, "ABU");
    }
    // --- Modes ---
    else if (command == "HAPPY") {
      setMode(MODE_HAPPY, "HAPPY");
    } else if (command == "SAD") {
      setMode(MODE_SAD, "SAD");
    } else if (command == "FIRE") { // Allow setting fire mode explicitly
        setMode(MODE_FIRE, "FIRE");
    } else if (command.indexOf("FRESH AIR") >= 0 || command.indexOf("HUMIDIFIER ON") >= 0 || command.indexOf("TURN ON HUMIDIFIER") >= 0) {
        setHumidifier(true);
        return;
    } else if (command.indexOf("STOP FRESH AIR") >= 0 || command.indexOf("HUMIDIFIER OFF") >= 0 || command.indexOf("TURN OFF HUMIDIFIER") >= 0 || command.indexOf("STOP HUMIDIFIER") >= 0) {
        setHumidifier(false);
        return;
    }
    // --- Unknown Command ---
    else {
      if (command.length() > 0) { // Only print if command is not empty
        Serial.print("Unknown command: ");
        Serial.println(command);
      }
      commandRecognized = false;
    }

    if (commandRecognized && command != "OFF" && command != "ON" && command.length() > 0) {
      // If a specific color or mode is set, ensure LEDs are turned on
      // (kecuali jika perintahnya adalah ON, karena itu sudah dihandle secara spesifik)
      ledsAreOn = true;
    }
  }
}

void setSolidColor(CRGB color, const char* colorName) {
  currentMode = MODE_SOLID_COLOR;
  solidColor = color;
  Serial.print("Mode set to SOLID_COLOR: ");
  Serial.println(colorName);
}

void setMode(LedMode mode, const char* modeName) {
  currentMode = mode;
  Serial.print("Mode set to: ");
  Serial.println(modeName);
}

// --- Mode Implementations ---

void runSolidColorMode() {
  fill_solid(leds, NUM_LEDS, solidColor);
}

void runFireMode() {
  // Array of temperature readings at each simulation cell
  static uint8_t heat[NUM_LEDS];

  // Step 1.  Cool down every cell a little
  for (int i = 0; i < NUM_LEDS; i++) {
    heat[i] = qsub8(heat[i], random8(0, ((COOLING * 10) / NUM_LEDS) + 2));
  }

  // Step 2.  Heat from each cell drifts 'up' and diffuses a little
  for (int k = NUM_LEDS - 1; k >= 2; k--) {
    heat[k] = (heat[k - 1] + heat[k - 2] + heat[k - 2]) / 3;
  }

  // Step 3.  Randomly ignite new 'sparks' of heat near the bottom
  if (random8() < SPARKING) {
    int y = random8(7); // Sparks ignite in the first 7 LEDs
    heat[y] = qadd8(heat[y], random8(160, 255));
  }

  // Step 4.  Map from heat cells to LED colors
  for (int j = 0; j < NUM_LEDS; j++) {
    CRGB color = HeatColor(heat[j]);
    int pixelnumber;
    if (gReverseDirection) {
      pixelnumber = (NUM_LEDS - 1) - j;
    } else {
      pixelnumber = j;
    }
    leds[pixelnumber] = color;
  }
}

void runHappyMode() {
  // Example: Rainbow cycle
  static uint8_t initialHue = 0;
  fill_rainbow(leds, NUM_LEDS, initialHue++, 7); // initialHue, deltaHue
  // EVERY_N_MILLISECONDS(20) { initialHue++; } // Or control speed like this
}

void runSadMode() {
  // Example: Slow pulsing blue
  // uint8_t brightness = beatsin8( BPM, MinBrightness, MaxBrightness, TimebaseOffset, PhaseOffset);
  // BPM: Beats Per Minute
  // MinBrightness, MaxBrightness: 0-255
  uint8_t sadBrightness = beatsin8(10, 30, 120); // Slow pulse (10 BPM), dim (30) to medium (120)
  fill_solid(leds, NUM_LEDS, CHSV(HUE_BLUE, 255, sadBrightness)); // Blue color, full saturation, varying brightness
}