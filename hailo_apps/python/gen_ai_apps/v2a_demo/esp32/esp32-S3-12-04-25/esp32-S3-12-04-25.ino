#define FASTLED_ALLOW_INTERRUPTS 0
#include <FastLED.h>

// Pin dan konfigurasi LED
#define DATA_PIN_DrLR    4       // Rear Left
#define DATA_PIN_DrLF    5      // Front Left
#define DATA_PIN_IP1     6       // IP Left
#define DATA_PIN_IP2     7       // IP Right
#define DATA_PIN_DrRF    15      // Front Right
#define DATA_PIN_DrRR    16      // Rear Right
#define COLOR_ORDER      GRB
#define LED_TYPE         WS2812B

#define NUM_LEDS_DrLR    72      // Rear Left count
#define NUM_LEDS_DrLF    75      // Front Left count
#define NUM_LEDS_IP1     10      // IP Left count
#define NUM_LEDS_IP2     45      // IP Right count
#define NUM_LEDS_DrRF    75      // Front Right count
#define NUM_LEDS_DrRR    72      // Rear Right count

#define NUM_LEDS (NUM_LEDS_DrLR + NUM_LEDS_DrLF + NUM_LEDS_IP1 + NUM_LEDS_IP2 + NUM_LEDS_DrRF + NUM_LEDS_DrRR)

// Hitung jumlah LED per sisi
const int leftTotal = NUM_LEDS_DrLR + NUM_LEDS_DrLF + NUM_LEDS_IP1;   // 72 + 75 + 10 = 157
const int rightTotal = NUM_LEDS_IP2 + NUM_LEDS_DrRF + NUM_LEDS_DrRR;    // 45 + 75 + 72 = 192

CRGB leds[NUM_LEDS];

// Warna untuk gradasi (dari biru ke merah)
CHSV start_hue = CHSV(160, 255, 100); // Biru
CHSV end_hue   = CHSV(0,   255, 100); // Merah

// Variabel mode
static int currentMode = 0;
CRGB selectedColor = CRGB::Black;
CRGB previousColor = CRGB::Black;

// Peta warna untuk perintah via serial
struct ColorMap {
  const char* name;
  CRGB color;
};

ColorMap colorMap[] = {
  { "merah",    CRGB::Red },
  { "hijau",    CRGB::Green },
  { "biru",     CRGB::Blue },
  { "magenta",  CRGB::Magenta },
  { "pink",     CRGB::Pink },
  { "violet",   CRGB::Violet },
  { "aqua",     CRGB::Aqua },
  { "kuning",   CRGB::Yellow },
  { "emas",     CRGB::Gold },
  { "navy",     CRGB::Navy },
  { "abu",      CRGB::Gray },
  { "lavender", CRGB::Lavender },
};

// Fungsi fade out untuk transisi smooth
void fadeOutTransition() {
  for (int i = 0; i < 20; i++) {
    fadeToBlackBy(leds, NUM_LEDS, 64);
    FastLED.show();
    delay(20);
  }
}


// Fungsi setMode untuk mengganti mode dengan animasi sesuai permintaan
void setMode(int mode) {
  currentMode = mode;
  // Clear LED sebelum mulai mode baru
  fill_solid(leds, NUM_LEDS, CRGB::Black);
  FastLED.show();

  if (mode == 0) {
    // Mode 0: Urutan nyala LED:
    // 1. IP: IP right (indeks 157–201) menyala dan IP left (indeks 147–156) mulai menyala saat 10 LED terakhir IP right.
    // 2. Front: Front right (indeks 202–276) menyala secara normal, front left (indeks 72–146) menyala dari indeks terkecil.
    // 3. Rear: Rear right (indeks 277–348) menyala normal, rear left (indeks 0–71) menyala dari indeks terkecil.
    
    // --- Animasi IP ---
    // IP Right: indeks 157-201 (45 LED)
    for (int i = 0; i < NUM_LEDS_IP2; i++) {
      int physIndexRight = 157 + i;
      uint8_t blendValRight = map(i, 0, rightTotal - 1, 0, 255);
      leds[physIndexRight] = blend(start_hue, end_hue, blendValRight);
      
      // IP Left: indeks 147-156 (10 LED) mulai saat LED IP Right telah mencapai 35 (45-10)
      if (i >= (NUM_LEDS_IP2 - NUM_LEDS_IP1)) {
        int j = i - (NUM_LEDS_IP2 - NUM_LEDS_IP1);  // j: 0 s/d 9
        int physIndexLeft = 147 + j;  // IP Left (urutan normal)
        uint8_t blendValLeft = map(j, 0, leftTotal - 1, 0, 255);
        leds[physIndexLeft] = blend(start_hue, end_hue, blendValLeft);
      }
      FastLED.show();
      delay(30);
    }
    
    // --- Animasi Front ---
    // Front Right: indeks 202-276 (75 LED) menyala normal, virtual index mulai 45
    // Front Left: indeks 72-146, menyala dari indeks terkecil (72 naik ke 146)
    for (int i = 0; i < NUM_LEDS_DrLF; i++) {
      // Front Right (normal)
      int physIndexRight = 202 + i;
      uint8_t blendValRight = map(45 + i, 0, rightTotal - 1, 0, 255);
      leds[physIndexRight] = blend(start_hue, end_hue, blendValRight);
      
      // Front Left (ascending order)
      int physIndexLeft = 72 + i;  // Mulai dari indeks terkecil (72) naik ke atas
      uint8_t blendValLeft = map(10 + i, 0, leftTotal - 1, 0, 255);
      leds[physIndexLeft] = blend(start_hue, end_hue, blendValLeft);
      
      FastLED.show();
      delay(30);
    }
    
    // --- Animasi Rear ---
    // Rear Right: indeks 277-348 (72 LED) menyala normal, virtual index mulai 120
    // Rear Left: indeks 0-71, menyala dari indeks terkecil (0 naik ke 71)
    for (int i = 0; i < NUM_LEDS_DrLR; i++) {
      // Rear Right (normal)
      int physIndexRight = 277 + i;
      uint8_t blendValRight = map(120 + i, 0, rightTotal - 1, 0, 255);
      leds[physIndexRight] = blend(start_hue, end_hue, blendValRight);
      
      // Rear Left (ascending order)
      int physIndexLeft = 0 + i;  // Mulai dari indeks terkecil (0) naik ke atas
      uint8_t blendValLeft = map(85 + i, 0, leftTotal - 1, 0, 255);
      leds[physIndexLeft] = blend(start_hue, end_hue, blendValLeft);
      
      FastLED.show();
      delay(30);
    }
    
  }
  else if (mode == 1) {
    // Matikan semua LED
    fill_solid(leds, NUM_LEDS, CRGB::Black);
    FastLED.show();
  }
  else if (mode == 2) {
    // Mode Rainbow: 6 section terpisah
    // Section 0: Rear Left (indeks 0–71)
    for (int i = 0; i < NUM_LEDS_DrLR; i++) {
      leds[i] = CHSV(map(i, 0, NUM_LEDS_DrLR - 1, 0, 255), 255, 255);
    }
    // Section 1: Front Left (72–146)
    for (int i = 0; i < NUM_LEDS_DrLF; i++) {
      leds[NUM_LEDS_DrLR + i] = CHSV(map(i, 0, NUM_LEDS_DrLF - 1, 0, 255), 255, 255);
    }
    // Section 2: IP Left (147–156)
    for (int i = 0; i < NUM_LEDS_IP1; i++) {
      leds[NUM_LEDS_DrLR + NUM_LEDS_DrLF + i] = CHSV(map(i, 0, NUM_LEDS_IP1 - 1, 0, 255), 255, 255);
    }
    // Section 3: IP Right (157–201)
    int startIP2 = NUM_LEDS_DrLR + NUM_LEDS_DrLF + NUM_LEDS_IP1;
    for (int i = 0; i < NUM_LEDS_IP2; i++) {
      leds[startIP2 + i] = CHSV(map(i, 0, NUM_LEDS_IP2 - 1, 0, 255), 255, 255);
    }
    // Section 4: Front Right (202–276)
    int startDrRF = startIP2 + NUM_LEDS_IP2;
    for (int i = 0; i < NUM_LEDS_DrRF; i++) {
      leds[startDrRF + i] = CHSV(map(i, 0, NUM_LEDS_DrRF - 1, 0, 255), 255, 255);
    }
    // Section 5: Rear Right (277–348)
    int startDrRR = startDrRF + NUM_LEDS_DrRF;
    for (int i = 0; i < NUM_LEDS_DrRR; i++) {
      leds[startDrRR + i] = CHSV(map(i, 0, NUM_LEDS_DrRR - 1, 0, 255), 255, 255);
    }
    FastLED.show();
  }
  else if (mode == 3) {
    // Mode biru redup
    fill_solid(leds, NUM_LEDS, CRGB(0, 0, 50));
    FastLED.show();
  }
  else if (mode == 4) {
    // Mode acak
    for (int i = 0; i < NUM_LEDS; i++) {
      leds[i] = CHSV(random(0, 255), 255, 255);
    }
    FastLED.show();
  }
}


#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#include <FastLED.h>

#ifdef __AVR__
#include <avr/power.h>
#endif


#define MAX_POWER_MILLIAMPS 500
#define FASTLED_ALLOW_INTERRUPTS 0
#define MAX_BRIGHTNESS 255
#define SUNRISE_LENGTH 30  // Waktu sunrise/sunset dalam detik
#define INTERVAL 50        // Delay antar perubahan warna
#define RISE 0
#define SET 1
#define DEFAULT_SPEED 0 // Kecepatan blinking default dalam milidetik
FASTLED_USING_NAMESPACE 

#define SERVICE_UUID "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
#define CHARACTERISTIC_UUID "beb5483e-36e1-4688-b7f5-ea07361b26a8"
#define bleServerName "ESP32 Gass"

BLEServer* pServer = NULL;
BLEService* pService = NULL;
BLECharacteristic* pCharacteristic = NULL;
BLEAdvertising* pAdvertising = NULL;

unsigned long previousMillis = 0;           // Stores last time LEDs were updated
int count = 0;                              // Stores count for incrementing up to the NUM_LEDs
bool deviceConnected = false;
bool isAnimationRunning = true;
uint8_t led_status = 0;
uint8_t led_animation = 37;
uint8_t led_animation_old = 0;
uint8_t led_brightness = 128;
uint8_t led_blinking = 500;
uint8_t led_red = 64;
uint8_t led_green = 64;
uint8_t led_blue = 64;


uint8_t red_led = 0;
uint8_t green_led = 0;
uint8_t blue_led = 0;
uint8_t brightness_led = 0;
uint8_t mode_led = 0;
uint8_t last_mode = 0;
bool stopAnimation = false; // Flag untuk menghentikan animasi
uint8_t choose = 0;



int sunDirection = RISE;
uint8_t heatIndex = 0;
unsigned long lastUpdate = 0;
int current_mode = 0;  // Menyimpan mode yang sedang aktif
unsigned long last_update = 0;  
const int interval = 50;  
int current_step = 1;  // Menyimpan progres Mode 1



int blinkSpeed = DEFAULT_SPEED;
bool ledState = false;
bool blinking = true;






//Function nyalakan dan matikan lampu
void changeLedStatus() {
  if (led_status == 0) {
    Serial.println("LED STATUS");
    fill_solid(leds, NUM_LEDS, CRGB::Black); 
    FastLED.show();    
  }
  else if(led_status == 1){
  
    fill_solid(leds, NUM_LEDS, CRGB::Red); 
    FastLED.show();    
  }
} 

//function brightness
void changeLedBrightness() {
  FastLED.setBrightness(led_brightness);
  FastLED.show();
}

//function blinking speed
// void blinkingSpeed(int delayTime) {
//   for (int i = 0; i < NUM_LEDS; i++) {
//     leds[i] = CRGB::White; // Nyalakan LED
//   }
//   FastLED.show();
//   delay(delayTime);

//   for (int i = 0; i < NUM_LEDS; i++) {
//     leds[i] = CRGB::Black; // Matikan LED
//   }
//   FastLED.show();
// }

// Color Picker
void singleColor(CRGB color) {
  fill_solid(leds, NUM_LEDS, color);
  FastLED.show();
}

void shootingStarAnimation(int red, int green, int blue, int tail_length, int delay_duration, int interval, int direction) {
    static unsigned long previousMillis = 0;
    static int count = 0;

    unsigned long currentMillis = millis();
    if (currentMillis - previousMillis >= interval) {
        previousMillis = currentMillis; 
        count = 0;  // Reset animasi
    }

    // Pastikan animasi terus berjalan
    if (count >= NUM_LEDS) {
        count = 0;
    }

    // Atur warna LED sesuai arah gerakan
    if (direction == -1) {
        leds[NUM_LEDS - count] = CRGB(red, green, blue);
    } else {
        leds[count] = CRGB(red, green, blue);
    }

    // Fade ekor bintang
    fadeToBlackBy(leds, NUM_LEDS, tail_length);
    
    FastLED.show();
    count++;  // Maju ke LED berikutnya
    delay(delay_duration);  // Atur kecepatan animasi
}

// Rainbow Animation
void rainbow() {
    static uint8_t j = 0;  // Variabel statis untuk menyimpan posisi warna

    for (int i = 0; i < NUM_LEDS; i++) {
        leds[i] = CHSV((i + j) % 255, 255, 255);
    }
    
    FastLED.show();
    j++; // Geser warna rainbow setiap pemanggilan fungsi
    delay(50); // Bisa diganti dengan millis() untuk non-blocking
}



// fire animation

void setPixelHeatColor(int Pixel, byte temperature) {
  // Rescale heat from 0-255 to 0-191
  byte t192 = round((temperature / 255.0) * 191);
  
  // Calculate ramp up from
  byte heatramp = t192 & 0x3F; // 0...63
  heatramp <<= 2; // scale up to 0...252
  
  // Figure out which third of the spectrum we're in:
  if(t192 > 0x80) {                    // hottest
    leds[Pixel].setRGB(255, 255, heatramp);
  }
  else if(t192 > 0x40) {               // middle
    leds[Pixel].setRGB(255, heatramp, 0);
  }
  else {                               // coolest
    leds[Pixel].setRGB(heatramp, 0, 0);
  }
}
void Fire(int FlameHeight, int Sparks, int DelayDuration) {
  static byte heat[NUM_LEDS];
  int cooldown;
  
  // Cool down each cell a little
  for(int i = 0; i < NUM_LEDS; i++) {
    cooldown = random(0, ((FlameHeight * 10) / NUM_LEDS) + 2);
   
    if(cooldown > heat[i]) {
      heat[i] = 0;
    }
    else {
      heat[i] = heat[i] - cooldown;
    }
  }
  
  // Heat from each cell drifts up and diffuses slightly
  for(int k = (NUM_LEDS - 1); k >= 2; k--) {
    heat[k] = (heat[k - 1] + heat[k - 2] + heat[k - 2]) / 3;
  }
  
  // Randomly ignite new Sparks near bottom of the flame
  if(random(255) < Sparks) {
    int y = random(7);
    heat[y] = heat[y] + random(160, 255);
  }
  
  // Convert heat to LED colors
  for(int j = 0; j < NUM_LEDS; j++) {
    setPixelHeatColor(j, heat[j]);
  }
  
  FastLED.show();
  delay(DelayDuration);
}

//animasi ocean
CRGBPalette16 pacifica_palette_1 = 
    { 0x000507, 0x000409, 0x00030B, 0x00030D, 0x000210, 0x000212, 0x000114, 0x000117, 
      0x000019, 0x00001C, 0x000026, 0x000031, 0x00003B, 0x000046, 0x14554B, 0x28AA50 };
CRGBPalette16 pacifica_palette_2 = 
    { 0x000507, 0x000409, 0x00030B, 0x00030D, 0x000210, 0x000212, 0x000114, 0x000117, 
      0x000019, 0x00001C, 0x000026, 0x000031, 0x00003B, 0x000046, 0x0C5F52, 0x19BE5F };
CRGBPalette16 pacifica_palette_3 = 
    { 0x000208, 0x00030E, 0x000514, 0x00061A, 0x000820, 0x000927, 0x000B2D, 0x000C33, 
      0x000E39, 0x001040, 0x001450, 0x001860, 0x001C70, 0x002080, 0x1040BF, 0x2060FF };


void pacifica_loop()
{
  // Increment the four "color index start" counters, one for each wave layer.
  // Each is incremented at a different speed, and the speeds vary over time.
  static uint16_t sCIStart1, sCIStart2, sCIStart3, sCIStart4;
  static uint32_t sLastms = 0;
  uint32_t ms = GET_MILLIS();
  uint32_t deltams = ms - sLastms;
  sLastms = ms;
  uint16_t speedfactor1 = beatsin16(3, 179, 269);
  uint16_t speedfactor2 = beatsin16(4, 179, 269);
  uint32_t deltams1 = (deltams * speedfactor1) / 256;
  uint32_t deltams2 = (deltams * speedfactor2) / 256;
  uint32_t deltams21 = (deltams1 + deltams2) / 2;
  sCIStart1 += (deltams1 * beatsin88(1011,10,13));
  sCIStart2 -= (deltams21 * beatsin88(777,8,11));
  sCIStart3 -= (deltams1 * beatsin88(501,5,7));
  sCIStart4 -= (deltams2 * beatsin88(257,4,6));

  // Clear out the LED array to a dim background blue-green
  fill_solid( leds, NUM_LEDS, CRGB( 2, 6, 10));

  // Render each of four layers, with different scales and speeds, that vary over time
  pacifica_one_layer( pacifica_palette_1, sCIStart1, beatsin16( 3, 11 * 256, 14 * 256), beatsin8( 10, 70, 130), 0-beat16( 301) );
  pacifica_one_layer( pacifica_palette_2, sCIStart2, beatsin16( 4,  6 * 256,  9 * 256), beatsin8( 17, 40,  80), beat16( 401) );
  pacifica_one_layer( pacifica_palette_3, sCIStart3, 6 * 256, beatsin8( 9, 10,38), 0-beat16(503));
  pacifica_one_layer( pacifica_palette_3, sCIStart4, 5 * 256, beatsin8( 8, 10,28), beat16(601));

  // Add brighter 'whitecaps' where the waves lines up more
  pacifica_add_whitecaps();

  // Deepen the blues and greens a bit
  pacifica_deepen_colors();
}

// Add one layer of waves into the led array
void pacifica_one_layer( CRGBPalette16& p, uint16_t cistart, uint16_t wavescale, uint8_t bri, uint16_t ioff)
{
  uint16_t ci = cistart;
  uint16_t waveangle = ioff;
  uint16_t wavescale_half = (wavescale / 2) + 20;
  for( uint16_t i = 0; i < NUM_LEDS; i++) {
    waveangle += 250;
    uint16_t s16 = sin16( waveangle ) + 32768;
    uint16_t cs = scale16( s16 , wavescale_half ) + wavescale_half;
    ci += cs;
    uint16_t sindex16 = sin16( ci) + 32768;
    uint8_t sindex8 = scale16( sindex16, 240);
    CRGB c = ColorFromPalette( p, sindex8, bri, LINEARBLEND);
    leds[i] += c;
  }
}

// Add extra 'white' to areas where the four layers of light have lined up brightly
void pacifica_add_whitecaps()
{
  uint8_t basethreshold = beatsin8( 9, 55, 65);
  uint8_t wave = beat8( 7 );
  
  for( uint16_t i = 0; i < NUM_LEDS; i++) {
    uint8_t threshold = scale8( sin8( wave), 20) + basethreshold;
    wave += 7;
    uint8_t l = leds[i].getAverageLight();
    if( l > threshold) {
      uint8_t overage = l - threshold;
      uint8_t overage2 = qadd8( overage, overage);
      leds[i] += CRGB( overage, overage2, qadd8( overage2, overage2));
    }
  }
}

// Deepen the blues and greens
void pacifica_deepen_colors()
{
  for( uint16_t i = 0; i < NUM_LEDS; i++) {
    leds[i].blue = scale8( leds[i].blue,  145); 
    leds[i].green= scale8( leds[i].green, 200); 
    leds[i] |= CRGB( 2, 5, 7);
  }
}

// Fungsi untuk transisi sunrise & sunset
void sunCycle() {
  if (sunDirection == RISE) {
    if (heatIndex < MAX_BRIGHTNESS) {
      heatIndex++;
    } else {
      sunDirection = SET;  // Setelah sunrise selesai, ganti ke sunset
      delay(5000);  // Tunggu 5 detik sebelum sunset dimulai
    }
  } 
  else if (sunDirection == SET) {
    if (heatIndex > 0) {
      heatIndex--;
    } else {
      sunDirection = RISE;  // Setelah sunset selesai, ganti ke sunrise
      delay(5000);  // Tunggu 5 detik sebelum sunrise dimulai
    }
  }

  // Warna gradasi sunrise/sunset
  fill_solid(leds, NUM_LEDS, CRGB(heatIndex, heatIndex / 2, 0));
  FastLED.show();
}

void Mode_RedToPurple() {
    static uint8_t hue = 0; // Mulai dari merah (hue 0)
    static bool increasing = true; // Arah perubahan hue

    if (millis() - last_update > interval) {
        last_update = millis();

        CHSV color(hue, 255, 255); // Warna dengan hue berubah
        fill_solid(leds, NUM_LEDS, color);
        FastLED.show();

        // Geser hue antara merah (0) dan ungu (170)
        if (increasing) {
            hue++;
            if (hue >= 170) increasing = false;
        } else {
            hue--;
            if (hue <= 0) increasing = true;
        }
    }
}


//Function menerima data animasi
void changeAnimation2() {
  
    switch (led_animation) {
        case 37:
            singleColor(CRGB(led_red, led_green, led_blue));
            break;

        case 36:
            shootingStarAnimation(led_red, led_green, led_blue, random(10, 96), random(5, 40), random(2000, 8000), 1);
            break;

        case 35:
            rainbow();
            break;

        case 34:
            Fire(55, 120, 15);
            break;

        case 33:
            EVERY_N_MILLISECONDS( 20) {
            pacifica_loop();
            FastLED.show();
            };
            break;

        case 32:
            if (millis() - lastUpdate > INTERVAL) { // Update setiap INTERVAL ms
              lastUpdate = millis();
            sunCycle();
            };
            break;

        case 31:
            Mode_RedToPurple(); // Warna merah
            break;
            

        default:
            singleColor(CRGB(led_red, led_green, led_blue));
            break;
    }
}

//function menerima data
class MyServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer* pServer) {
    deviceConnected = true;
    Serial.println("BLE: connected!");
  };

  void onDisconnect(BLEServer* pServer) {
    deviceConnected = false;
    Serial.println("BLE: disconnected!");
    BLEDevice::startAdvertising();
  }
};

class CharacteristicCallback : public BLECharacteristicCallbacks {
  void onWrite(BLECharacteristic* pCharacteristic) {
   uint8_t* received_data = pCharacteristic->getData();
int length = pCharacteristic->getLength(); // Ambil panjang data

Serial.print("Received Data: ");
for (int i = 0; i < length; i++) {
    Serial.print(received_data[i], HEX); // Cetak dalam format HEX
    Serial.print(" ");
}
Serial.println();
    //nyalakan matikan led
 if (received_data[0] == 2) {
      led_status = received_data[1];
    //Pemilihan Animasi
    } else if (received_data[0] == 9) {
      blinking = false;
      isAnimationRunning = true;
      led_animation = received_data[1];
      if (led_animation != led_animation_old) {
        led_animation_old = led_animation;
        isAnimationRunning = false;
        delay(20);
        isAnimationRunning = true;
      }
    //Color Picker
    } else if (received_data[0] == 37) {
      led_red = received_data[1];
      led_green = received_data[2];
      led_blue = received_data[3];
    } else if (received_data[0] == 51) { // **Blinking Speed**
        int speed = received_data[1]; // Sesuaikan nilai speed
        if (speed == 0) {
            blinking = false;
            Serial.println("Blinking OFF");
        } else if (speed >= 50 && speed <= 2000) {
            blinkSpeed = speed;
            blinking = true;
            Serial.printf("Blinking Speed diubah ke: %d ms\n", blinkSpeed);
        }
    } else if (received_data[0] == 32) {
      led_brightness = received_data[1];
    //??
    } else if (received_data[0] == 48) {
      if (received_data[1] == 1) {
        sendInfo();
      }
}else if (received_data[0] == 80) { // Cek Start Byte
    stopAnimation = false; // Flag untuk menghentikan animasi
    red_led = received_data[1];
    green_led  = received_data[2];
    blue_led  = received_data[3];
    brightness_led = received_data[4];
    mode_led = received_data[5];

    Serial.println("Received Data:");
    Serial.print("Red: "); Serial.println(red_led);
    Serial.print("Green: "); Serial.println(green_led);
    Serial.print("Blue: "); Serial.println(blue_led);
    Serial.print("Brightness: "); Serial.println(brightness_led);
    Serial.print("Mode: "); Serial.println(mode_led);

}else if (received_data[0] == 81){
  stopAnimation = true; 

  


    if (!stopAnimation) { 
        if (mode_led != last_mode) {
            last_mode = mode_led; // Perbarui mode terakhir
        }

        switch (mode_led) {
            // case 51: FadeInOut(red_led, green_led, blue_led, brightness_led); break;
            // case 52: Pulse(red_led, green_led, blue_led, brightness_led); break;
            // case 53: Strobe(red_led, green_led, blue_led, brightness_led); break;
            case 54: Wipe(red_led, green_led, blue_led, brightness_led); break;
            default:
                Serial.println("Mode tidak dikenali");
                break;
        }
    } else {
      CHSV(0, 0, 0);
    }

}


    changeLedStatus();
    changeLedBrightness();
  }
};



//Function BLE
void setupBle() {
  Serial.println("BLE initializing...");
  BLEDevice::init(bleServerName);
  pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());
  pService = pServer->createService(SERVICE_UUID);
  pCharacteristic = pService->createCharacteristic(
    CHARACTERISTIC_UUID,
    BLECharacteristic::PROPERTY_NOTIFY |
    BLECharacteristic::PROPERTY_READ |
    BLECharacteristic::PROPERTY_WRITE);
  pCharacteristic->addDescriptor(new BLE2902());
  pCharacteristic->setCallbacks(new CharacteristicCallback());
  pService->start();
  pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->addServiceUUID(SERVICE_UUID);
  pAdvertising->setScanResponse(true);
  pAdvertising->setMinPreferred(0x06);
  pAdvertising->setMinPreferred(0x12);
  BLEDevice::startAdvertising();
  Serial.println("BLE initialized. Waiting for client to connect...");
}
void sendInfo() {
  uint8_t ledInfoArr[] = { led_status, led_animation, led_brightness, led_red, led_green, led_blue, led_blinking };
  pCharacteristic->setValue(ledInfoArr, sizeof(ledInfoArr));
}







// === Transisi 1: Fade In-Out ===
void FadeInOut(byte red, byte green, byte blue, byte brightness) {
  float r, g, b;
  for (int k = 0; k < 256; k += 5) {
    if (mode_led != 51) return; // Keluar jika mode berubah
    float scale = (k / 256.0) * (brightness / 255.0);
    r = red * scale;
    g = green * scale;
    b = blue * scale;
    setAll(r, g, b);
    FastLED.show();
    delay(10);
  }
  for (int k = 255; k >= 0; k -= 5) {
    if (mode_led != 51) return;
    float scale = (k / 256.0) * (brightness / 255.0);
    r = red * scale;
    g = green * scale;
    b = blue * scale;
    setAll(r, g, b);
    FastLED.show();
    delay(10);
  }
}

// === Transisi 2: Pulse ===
void Pulse(byte red, byte green, byte blue, byte brightness) {
  for (int k = 0; k < 256; k += 10) {
    if (mode_led != 52) return;
    float scale = sin(k * PI / 256.0) * (brightness / 255.0);
    setAll(red * scale, green * scale, blue * scale);
    FastLED.show();
    delay(20);
  }
}

// === Transisi 3: Strobe ===
void Strobe(byte red, byte green, byte blue, byte brightness) {
  for (int i = 0; i < 10; i++) {
    if (mode_led != 53) return;
    setAll(red * (brightness / 255.0), green * (brightness / 255.0), blue * (brightness / 255.0));
    FastLED.show();
    delay(100);
    setAll(0, 0, 0);
    FastLED.show();
    delay(100);
  }
}


void Wipe(byte red, byte green, byte blue, byte brightness) {
  if (mode_led != 54) return;
  
  // Konversi brightness
  red = red * (brightness / 255.0);
  green = green * (brightness / 255.0);
  blue = blue * (brightness / 255.0);
  
  // --- Animasi IP (Reverse) ---
  for (int i = NUM_LEDS_IP2 - 1; i >= 0; i--) {
    int physIndexRight = 157 + i;
    uint8_t blendValRight = map(i, 0, rightTotal - 1, 0, 255);
    leds[physIndexRight] = blend(CRGB(red, green, blue), CRGB(red, green, blue), blendValRight);
    
    if (i >= (NUM_LEDS_IP2 - NUM_LEDS_IP1)) {
      int j = i - (NUM_LEDS_IP2 - NUM_LEDS_IP1);
      int physIndexLeft = 147 + j;
      uint8_t blendValLeft = map(j, 0, leftTotal - 1, 0, 255);
      leds[physIndexLeft] = blend(CRGB(red, green, blue), CRGB(red, green, blue), blendValLeft);
    }
    FastLED.show();
    delay(30);
  }

  // --- Animasi Front (Reverse) ---
  for (int i = NUM_LEDS_DrLF - 1; i >= 0; i--) {
    int physIndexRight = 202 + i;
    uint8_t blendValRight = map(45 + i, 0, rightTotal - 1, 0, 255);
    leds[physIndexRight] = blend(CRGB(red, green, blue), CRGB(red, green, blue), blendValRight);
    
    int physIndexLeft = 72 + i;
    uint8_t blendValLeft = map(10 + i, 0, leftTotal - 1, 0, 255);
    leds[physIndexLeft] = blend(CRGB(red, green, blue), CRGB(red, green, blue), blendValLeft);
    
    FastLED.show();
    delay(30);
  }
  
  // --- Animasi Rear (Reverse) ---
  for (int i = NUM_LEDS_DrLR - 1; i >= 0; i--) {
    int physIndexRight = 277 + i;
    uint8_t blendValRight = map(120 + i, 0, rightTotal - 1, 0, 255);
    leds[physIndexRight] = blend(CRGB(red, green, blue), CRGB(red, green, blue), blendValRight);
    
    int physIndexLeft = 0 + i;
    uint8_t blendValLeft = map(85 + i, 0, leftTotal - 1, 0, 255);
    leds[physIndexLeft] = blend(CRGB(red, green, blue), CRGB(red, green, blue), blendValLeft);
    
    FastLED.show();
    delay(30);
  }

  //   // --- Animasi IP ---
  // for (int i = 0; i < NUM_LEDS_IP2; i++) {
  //   int physIndexRight = 157 + i;
  //   uint8_t blendValRight = map(i, 0, rightTotal - 1, 0, 255);
  //   leds[physIndexRight] = blend(CRGB(red, green, blue), CRGB(red, green, blue), blendValRight);
    
  //   if (i >= (NUM_LEDS_IP2 - NUM_LEDS_IP1)) {
  //     int j = i - (NUM_LEDS_IP2 - NUM_LEDS_IP1);
  //     int physIndexLeft = 147 + j;
  //     uint8_t blendValLeft = map(j, 0, leftTotal - 1, 0, 255);
  //     leds[physIndexLeft] = blend(CRGB(red, green, blue), CRGB(red, green, blue), blendValLeft);
  //   }
  //   FastLED.show();
  //   delay(30);
  // }

  // // --- Animasi Front ---
  // for (int i = 0; i < NUM_LEDS_DrLF; i++) {
  //   int physIndexRight = 202 + i;
  //   uint8_t blendValRight = map(45 + i, 0, rightTotal - 1, 0, 255);
  //   leds[physIndexRight] = blend(CRGB(red, green, blue), CRGB(red, green, blue), blendValRight);
    
  //   int physIndexLeft = 72 + i;
  //   uint8_t blendValLeft = map(10 + i, 0, leftTotal - 1, 0, 255);
  //   leds[physIndexLeft] = blend(CRGB(red, green, blue), CRGB(red, green, blue), blendValLeft);
    
  //   FastLED.show();
  //   delay(30);
  // }
  
  // // --- Animasi Rear ---
  // for (int i = 0; i < NUM_LEDS_DrLR; i++) {
  //   int physIndexRight = 277 + i;
  //   uint8_t blendValRight = map(120 + i, 0, rightTotal - 1, 0, 255);
  //   leds[physIndexRight] = blend(CRGB(red, green, blue), CRGB(red, green, blue), blendValRight);
    
  //   int physIndexLeft = 0 + i;
  //   uint8_t blendValLeft = map(85 + i, 0, leftTotal - 1, 0, 255);
  //   leds[physIndexLeft] = blend(CRGB(red, green, blue), CRGB(red, green, blue), blendValLeft);
    
  //   FastLED.show();
  //   delay(30);
  // }
}





// === Helper Functions ===
void setAll(byte red, byte green, byte blue) {
  for (int i = 0; i < NUM_LEDS; i++) {
    setPixel(i, red, green, blue);
  }
  FastLED.show();
}

void setPixel(int Pixel, byte red, byte green, byte blue) {
  leds[Pixel].r = red;
  leds[Pixel].g = green;
  leds[Pixel].b = blue;
}






//////////////////////////////////////
// Setup dan Loop
//////////////////////////////////////
void setup() {
  // Konfigurasi LED per segmen
  FastLED.addLeds<LED_TYPE, DATA_PIN_DrLR, COLOR_ORDER>(leds, NUM_LEDS_DrLR);
  FastLED.addLeds<LED_TYPE, DATA_PIN_DrLF, COLOR_ORDER>(leds + NUM_LEDS_DrLR, NUM_LEDS_DrLF);
  FastLED.addLeds<LED_TYPE, DATA_PIN_IP1,  COLOR_ORDER>(leds + NUM_LEDS_DrLR + NUM_LEDS_DrLF, NUM_LEDS_IP1);
  FastLED.addLeds<LED_TYPE, DATA_PIN_IP2,  COLOR_ORDER>(leds + NUM_LEDS_DrLR + NUM_LEDS_DrLF + NUM_LEDS_IP1, NUM_LEDS_IP2);
  FastLED.addLeds<LED_TYPE, DATA_PIN_DrRF, COLOR_ORDER>(leds + NUM_LEDS_DrLR + NUM_LEDS_DrLF + NUM_LEDS_IP1 + NUM_LEDS_IP2, NUM_LEDS_DrRF);
  FastLED.addLeds<LED_TYPE, DATA_PIN_DrRR, COLOR_ORDER>(leds + NUM_LEDS_DrLR + NUM_LEDS_DrLF + NUM_LEDS_IP1 + NUM_LEDS_IP2 + NUM_LEDS_DrRF, NUM_LEDS_DrRR);
  

  FastLED.setMaxPowerInVoltsAndMilliamps( 5, MAX_POWER_MILLIAMPS);
  #if defined(__AVR_ATtiny85__)
  if (F_CPU == 16000000) clock_prescale_set(clock_div_1);
  #endif
  Serial.begin(115200);
  changeLedBrightness();
  FastLED.setBrightness(MAX_BRIGHTNESS);
  // blinkingSpeed(led_blinking);
    // Set mode awal, misal mode 0
  setMode(0);
  setupBle();
  FastLED.show();//
}

// void loop() {
//   // Baca perintah Serial untuk mengganti mode/warna
//   if (Serial.available()) {
//     String command = Serial.readStringUntil('\n');
//     command.trim();
    
//     // Transisi fade out sebelum pergantian mode/warna
//     fadeOutTransition();
    
//     if (command.equalsIgnoreCase("ON")) {
//       setMode(0);
//       Serial.println("LIGHTING: ON (Mode 0)");
//     }
//     else if (command.equalsIgnoreCase("OFF")) {
//       setMode(1);
//       Serial.println("LIGHTING: OFF");
//     }
//     else if (command.equalsIgnoreCase("HAPPY")) {
//       setMode(2);
//       Serial.println("MODE: RAINBOW");
//     }
//     else if (command.equalsIgnoreCase("SAD")) {
//       setMode(3);
//       Serial.println("MODE: BLUE");
//     }
//     else if (command.equalsIgnoreCase("ACAK")) {
//       setMode(4);
//       Serial.println("MODE: RANDOM");
//     }
//     else {
//       bool colorFound = false;
//       for (auto& entry : colorMap) {
//         if (command.equalsIgnoreCase(entry.name)) {
//           fadeOutTransition();
//           selectedColor = entry.color;
//           fill_solid(leds, NUM_LEDS, selectedColor);
//           FastLED.show();
//           Serial.print("COLOR: ");
//           Serial.println(entry.name);
//           previousColor = selectedColor;
//           colorFound = true;
//           break;
//         }
//       }
//       if (!colorFound) {
//         Serial.println("COMMAND ERROR");
//       }
//     }
//   }


//       unsigned long currentMillis = millis(); // Ambil waktu sekarang

//     if (led_status) { 
//         if (isAnimationRunning) { // Matikan blinking saat animasi berjalan
//             changeAnimation2(); // Jalankan animasi
//         }if (blinking && currentMillis - previousMillis >= blinkSpeed) { 
//             previousMillis = currentMillis; // Simpan waktu blinking terakhir
//             ledState = !ledState; // Toggle LED

//             if (ledState) {
//                 fill_solid(leds, NUM_LEDS, CRGB(led_red, led_green, led_blue)); // LED nyala
//             } else {
//                 fill_solid(leds, NUM_LEDS, CRGB::Black); // LED mati
//             }
//             FastLED.show(); // Tampilkan perubahan ke LED
//         }
//     }
//   if (!stopAnimation) { 
//         if (mode_led != last_mode) {
//             last_mode = mode_led; // Perbarui mode terakhir
//         }

//         switch (mode_led) {
//             case 51: FadeInOut(red_led, green_led, blue_led, brightness_led); break;
//             case 52: Pulse(red_led, green_led, blue_led, brightness_led); break;
//             case 53: Strobe(red_led, green_led, blue_led, brightness_led); break;
//             case 54: Wipe(red_led, green_led, blue_led, brightness_led); break;
//             default:
//                 Serial.println("Mode tidak dikenali");
//                 break;
//         }
//     } else {
//         CHSV(0, 0, 0); // Fungsi untuk mematikan LED saat animasi dihentikan
//     }

//       if (deviceConnected) { // Jika ada perangkat BLE yang terhubung
//         sendInfo();
//         delay(10);
//     }
      
// }

void loop() {
  if (deviceConnected) { // Jika perangkat BLE terhubung
    sendInfo();
    delay(10);
    
    unsigned long currentMillis = millis(); // Ambil waktu sekarang

    if (led_status) { 
      if (isAnimationRunning) { // Matikan blinking saat animasi berjalan
        changeAnimation2(); // Jalankan animasi
      }
      if (blinking && currentMillis - previousMillis >= blinkSpeed) { 
        previousMillis = currentMillis; // Simpan waktu blinking terakhir
        ledState = !ledState; // Toggle LED

        if (ledState) {
          fill_solid(leds, NUM_LEDS, CRGB(led_red, led_green, led_blue)); // LED nyala
        } else {
          fill_solid(leds, NUM_LEDS, CRGB::Black); // LED mati
        }
        FastLED.show(); // Tampilkan perubahan ke LED
      }
    }

    if (!stopAnimation) { 
      if (mode_led != last_mode) {
        last_mode = mode_led; // Perbarui mode terakhir
      }

      switch (mode_led) {
        case 51: FadeInOut(red_led, green_led, blue_led, brightness_led); break;
        case 52: Pulse(red_led, green_led, blue_led, brightness_led); break;
        case 53: Strobe(red_led, green_led, blue_led, brightness_led); break;
        case 54: Wipe(red_led, green_led, blue_led, brightness_led); break;
        default:

          break;
      }
    } else {
      FastLED.show();
    }
  } 
  else { // Jika perangkat BLE tidak terhubung, baca input dari Serial
    if (Serial.available()) {
      String command = Serial.readStringUntil('\n');
      command.trim();

      // Transisi fade out sebelum pergantian mode/warna
      fadeOutTransition();

      if (command.equalsIgnoreCase("ON")) {
        setMode(0);
        Serial.println("LIGHTING: ON (Mode 0)");
      }
      else if (command.equalsIgnoreCase("OFF")) {
        setMode(1);
        Serial.println("LIGHTING: OFF");
      }
      else if (command.equalsIgnoreCase("HAPPY")) {
        setMode(2);
        Serial.println("MODE: RAINBOW");
      }
      else if (command.equalsIgnoreCase("SAD")) {
        setMode(3);
        Serial.println("MODE: BLUE");
      }
      else if (command.equalsIgnoreCase("ACAK")) {
        setMode(4);
        Serial.println("MODE: RANDOM");
      }
      else {
        bool colorFound = false;
        for (auto& entry : colorMap) {
          if (command.equalsIgnoreCase(entry.name)) {
            fadeOutTransition();
            selectedColor = entry.color;
            fill_solid(leds, NUM_LEDS, selectedColor);
            FastLED.show();
            Serial.print("COLOR: ");
            Serial.println(entry.name);
            previousColor = selectedColor;
            colorFound = true;
            break;
          }
        }
        if (!colorFound) {
          Serial.println("COMMAND ERROR");
        }
      }
    }
  }
}
