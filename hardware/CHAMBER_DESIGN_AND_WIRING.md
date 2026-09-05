# 📷 LeafLens AI Pro: Scanning Chamber Design & Wiring Specification
**Lead Engineer:** Rachi Gedam (Hardware & IoT Specialist)  
**Target Hardware:** Raspberry Pi 4 Model B / Raspberry Pi Zero 2 W / Raspberry Pi 5  

---

## 1. Physical Scanning Chamber Architecture

### 📐 Enclosure Dimensions:
* **Width:** $20\text{ cm}$
* **Depth:** $20\text{ cm}$
* **Height:** $25\text{ cm}$
* **Material:** 3mm to 5mm Matte-Black Acrylic Sheets OR 3D-printed PLA.
* **Interior Lining:** Ultra-matte black chart paper or flocking tape to eliminate light reflections.

### 💡 Illumination & Optics:
1. **Camera Position:** Top-mounted at a fixed focal distance ($18\text{ cm}\text{–}20\text{ cm}$) pointing downward.
2. **Camera Sensor:** **Raspberry Pi Camera Module 3** (12MP Sony IMX708 with Autofocus and HDR) or **Pi HQ Camera** with 6mm CS-mount lens.
3. **Lighting Geometry:** Circular **CRI 95+ Daylight White LED Ring (5500K)** mounted around the lens with a translucent acrylic diffuser. The light hits the leaf at a $45^\circ$ angle, preventing specular glare on the waxy cuticle of grapevine leaves.
4. **Leaf Stage:** Flat non-reflective neutral-gray or matte-black tray at the bottom with a slide-out drawer mechanism.

---

## 2. Sensor Suite & Pinout Connection Diagram

| Component | Interface | Raspberry Pi Physical Pin | BCM GPIO Pin | Notes |
|---|---|---|---|---|
| **Sensirion SHT31 (VCC)** | Power | **Pin 1** (3.3V) | - | Sensor Power |
| **Sensirion SHT31 (GND)** | Ground | **Pin 6** (GND) | - | Ground |
| **Sensirion SHT31 (SDA)** | $I^2C$ Data | **Pin 3** | GPIO 2 | Shared $I^2C$ bus |
| **Sensirion SHT31 (SCL)** | $I^2C$ Clock | **Pin 5** | GPIO 3 | Shared $I^2C$ bus |
| **BH1750 Lux (VCC)** | Power | **Pin 17** (3.3V) | - | Sensor Power |
| **BH1750 Lux (GND)** | Ground | **Pin 9** (GND) | - | Ground |
| **BH1750 Lux (SDA)** | $I^2C$ Data | **Pin 3** | GPIO 2 | Connected to same SDA |
| **BH1750 Lux (SCL)** | $I^2C$ Clock | **Pin 5** | GPIO 3 | Connected to same SCL |
| **Tactile Push Button** | Digital In | **Pin 11** | GPIO 17 | Pull-Up enabled (press to GND) |
| **Status Green LED** | Digital Out | **Pin 13** | GPIO 27 | In series with $330\ \Omega$ resistor |
| **Pi Camera Module 3** | MIPI CSI | **CSI Port** | - | 15-pin ribbon cable |

---

## 3. Bill of Materials (BOM) & Estimated Budget

| Item | Component Description | Approx Cost (₹ INR) | Status |
|---|---|:---:|:---:|
| 1 | **Raspberry Pi 4 Model B (2GB/4GB)** or **Zero 2 W** | ₹3,500 – ₹5,500 | Required |
| 2 | **Raspberry Pi Camera Module 3 (Wide/Autofocus)** | ₹2,400 | Required |
| 3 | **Sensirion SHT31-D Temp & Humidity Sensor Module** | ₹450 | Procure |
| 4 | **BH1750 Ambient Light Sensor Module** | ₹150 | Procure |
| 5 | **5V CRI 95+ USB LED Ring Light with Diffuser** | ₹400 | Procure |
| 6 | **Matte Black Acrylic / 3D Printed Box Enclosure** | ₹800 | Fabricate |
| 7 | **MicroSD Card (32GB Class 10 A1)** | ₹400 | Available |
| 8 | **Push Button, Green LED, 330Ω Resistor & Jumper Wires** | ₹150 | Available |
| **TOTAL** | **Estimated Prototype Cost** | **~₹8,250 – ₹10,250** | |

---

## 4. Raspberry Pi Software Setup Instructions (For Rachi)

On the Raspberry Pi OS terminal, run:

```bash
# 1. Enable Camera and I2C interfaces
sudo raspi-config
# Navigate to: Interface Options -> I2C -> Enable -> Yes
# Navigate to: Interface Options -> Camera -> Enable -> Yes

# 2. Install required Python packages
sudo apt update
sudo apt install -y python3-pip python3-smbus i2c-tools
pip3 install requests smbus2 RPi.GPIO

# 3. Verify I2C sensors are detected
i2cdetect -y 1
# You should see: 0x44 (SHT31) and 0x23 (BH1750)

# 4. Set Gateway Server IP and Run Edge Node
export LEAFLENS_SERVER_URL="http://<YOUR_LAPTOP_IP>:8000/api/v1/scan"
python3 hardware/capture_node.py
```
