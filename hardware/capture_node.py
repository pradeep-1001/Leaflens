"""
LeafLens AI Pro - Raspberry Pi Edge Capture Node
Author: Rachi Gedam (Hardware & IoT Specialist)
Target Hardware: Raspberry Pi 4B / Zero 2W / Pi 5
Sensors: Pi Camera Module 3, Sensirion SHT31, BH1750, Push Button, Status LED
"""

import time
import os
import sys
import requests

# Reconfigure stdout for UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Configuration
SERVER_URL = os.environ.get("LEAFLENS_SERVER_URL", "http://127.0.0.1:8000/api/v1/scan")
DEVICE_ID = "RPI-CHAMBER-NIPHAD-01"
BUTTON_PIN = 17    # Physical GPIO pin for tactile scan button
STATUS_LED_PIN = 27 # Green indicator LED

# Check if running on actual Raspberry Pi hardware
IS_RASPBERRY_PI = False
try:
    import RPi.GPIO as GPIO
    import smbus2
    IS_RASPBERRY_PI = True
    print("[*] Running on Raspberry Pi Hardware Environment")
except (ImportError, RuntimeError):
    print("[!] Non-Raspberry Pi environment detected. Running in Edge Simulation Mode.")

def init_hardware():
    if IS_RASPBERRY_PI:
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(STATUS_LED_PIN, GPIO.OUT)
        GPIO.output(STATUS_LED_PIN, GPIO.LOW)
        print("[OK] GPIO pins configured.")

def read_sht31_sensor():
    """Reads Temperature & Humidity from Sensirion SHT31 (I2C 0x44)"""
    if not IS_RASPBERRY_PI:
        # Realistic simulation of Niphad, Nashik microclimate
        return 24.8, 84.5
    try:
        bus = smbus2.SMBus(1)
        # Send high repeatability measurement command
        bus.write_i2c_block_data(0x44, 0x24, [0x00])
        time.sleep(0.02)
        data = bus.read_i2c_block_data(0x44, 0x00, 6)
        temp_c = -45 + (175 * ((data[0] << 8) + data[1]) / 65535.0)
        humidity = 100 * ((data[3] << 8) + data[4]) / 65535.0
        return round(temp_c, 2), round(humidity, 2)
    except Exception as e:
        print(f"[!] Warning: SHT31 sensor read error ({e}). Using default values.")
        return 24.0, 80.0

def read_bh1750_lux():
    """Reads Ambient Illumination in Lux from BH1750 (I2C 0x23)"""
    if not IS_RASPBERRY_PI:
        return 850.0
    try:
        bus = smbus2.SMBus(1)
        data = bus.read_i2c_block_data(0x23, 0x10, 2)
        lux = ((data[0] << 8) + data[1]) / 1.2
        return round(lux, 1)
    except Exception:
        return 850.0

def capture_leaf_image(output_path="leaf_capture.jpg"):
    """Captures leaf image via Pi Camera or standard USB/webcam"""
    print("[+] Triggering camera capture...")
    if IS_RASPBERRY_PI:
        # Use native Raspberry Pi libcamera command
        os.system(f"libcamera-still -t 500 -o {output_path} --width 1024 --height 1024 -n")
    else:
        # Fallback for PC testing: use OpenCV
        try:
            import cv2
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(output_path, frame)
            cap.release()
        except Exception:
            pass

    if not os.path.exists(output_path):
        print(f"[!] Warning: Camera not found. Creating placeholder test image.")
        from PIL import Image
        img = Image.new('RGB', (224, 224), color=(34, 139, 34))
        img.save(output_path)

    return output_path

def trigger_scan_and_transmit():
    print("\n" + "=" * 50)
    print("🍇 LEAFLENS EDGE SCAN INITIATED")
    print("=" * 50)

    # 1. Read Environmental Sensors
    temp, rh = read_sht31_sensor()
    lux = read_bh1750_lux()
    leaf_wetness = (rh > 80.0) # High humidity indicates leaf surface dew
    print(f"[*] Chamber Microclimate: Temp={temp}°C | RH={rh}% | Lux={lux} | Wetness={leaf_wetness}")

    # 2. Capture Image
    img_path = os.path.join(os.path.dirname(__file__), "leaf_sample.jpg")
    capture_leaf_image(img_path)

    # 3. Transmit to Gargi's Backend API
    print(f"[*] Transmitting to LeafLens Gateway: {SERVER_URL}...")
    payload = {
        "device_id": DEVICE_ID,
        "temperature_c": str(temp),
        "humidity_pct": str(rh),
        "lux": str(lux),
        "leaf_wetness": str(leaf_wetness).lower()
    }

    try:
        with open(img_path, "rb") as img_file:
            files = {"image": ("leaf.jpg", img_file, "image/jpeg")}
            response = requests.post(SERVER_URL, data=payload, files=files, timeout=15)
            
        if response.status_code == 200:
            res_data = response.json()
            print("\n✅ DIAGNOSIS RECEIVED FROM GATEWAY:")
            print(f"  • Diagnosis   : {res_data.get('disease_diagnosis')}")
            print(f"  • Confidence  : {res_data.get('confidence') * 100:.1f}%")
            print(f"  • Health Score: {res_data.get('plant_health_score')} / 100")
            print(f"  • Risk Status : {res_data.get('microclimate', {}).get('risk_status')}")
            print("  • Advisory    :")
            for rec in res_data.get('recommendations', []):
                print(f"    - {rec}")

            # Blink LED on success
            if IS_RASPBERRY_PI:
                GPIO.output(STATUS_LED_PIN, GPIO.HIGH)
                time.sleep(1)
                GPIO.output(STATUS_LED_PIN, GPIO.LOW)
        else:
            print(f"[!] Gateway Error [{response.status_code}]: {response.text}")
    except Exception as e:
        print(f"[❌] Network Error connecting to gateway: {e}")

def run_loop():
    init_hardware()
    print("\n🚀 LeafLens Edge Node Ready! Listening for button triggers...")
    print("Press Ctrl+C to stop.\n")

    if not IS_RASPBERRY_PI:
        # Run one single test scan in simulation mode
        trigger_scan_and_transmit()
        return

    try:
        while True:
            # Detect tactile button press (active low)
            if GPIO.input(BUTTON_PIN) == GPIO.LOW:
                print("\n[🔘] Hardware push-button pressed!")
                trigger_scan_and_transmit()
                time.sleep(2) # Debounce delay
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping Edge Node...")
    finally:
        if IS_RASPBERRY_PI:
            GPIO.cleanup()

if __name__ == "__main__":
    run_loop()
