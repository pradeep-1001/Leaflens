# 🍇 LeafLens AI Pro: Complete Project Documentation & Evaluation Kit

**Project Title:** LeafLens AI Pro: AI-Based Grapevine Disease Detection & Smart Health Monitoring  
**Target Crop:** Grapevine (*Vitis vinifera*)  
**Geographical Focus:** Niphad, Nashik Grape Belt (Maharashtra, India)  
**Core Technologies:** PyTorch • MobileNetV3 • Computer Vision • IoT • Raspberry Pi • FastAPI • Streamlit • Agronomic Analytics  

---

## 👥 Team Roles & Verified Deliverables

| Member | Focus Area | Key Output & Verified Artifact |
|---|---|---|
| **Jiya Gavali** | **AI & Disease Detection** | Trained PyTorch model (`saved_models/best_model.pth`), Grad-CAM heatmap engine (`inference.py`), 98.5% validation accuracy benchmark (`evaluate_model.py`). |
| **Rachi Gedam** | **Hardware & IoT** | Raspberry Pi edge capture node (`hardware/capture_node.py`), chamber dimensions, wiring pinouts, and sensor BOM (`hardware/CHAMBER_DESIGN_AND_WIRING.md`). |
| **Gargi Shinde** | **Backend & Integration** | FastAPI REST Gateway (`backend/server.py`), persistent SQLite database (`backend/database.py`), multi-modal environmental fusion logic & ICAR-NRCG advisory engine. |
| **Gauri Patil** | **Dashboard & UI** | Interactive web dashboard (`app.py`), live camera input, side-by-side Grad-CAM inspector, 30-day temporal trend charts, bilingual English & Marathi localization. |

---

## 🏛️ End-to-End System Architecture

```text
[Raspberry Pi Scanning Chamber (Rachi)]
  ├── Diffused CRI 95+ LED Ring Light (Glare-Free)
  ├── Sensirion SHT31 (Temp & Humidity) + BH1750 (Lux)
  └── Pi Camera Module 3 (Sony IMX708 Autofocus)
             │
             │ HTTP POST /api/v1/scan (Multipart Image + JSON Sensors)
             ▼
[FastAPI Gateway Server (Gargi)]
  ├── Step 1: Ingests payload & stores image to /outputs/
  ├── Step 2: Calls PyTorch MobileNetV3-Large Classifier (Jiya)
  ├── Step 3: Generates Grad-CAM Visual Attention Heatmap
  ├── Step 4: Executes Multi-Modal Environmental Risk Fusion
  ├── Step 5: Computes Plant Health Score (0-100) & VPD (kPa)
  ├── Step 6: Generates ICAR-NRCG Aligned Treatment Advisory
  └── Step 7: Logs record to SQLite Database (leaflens.db)
             │
             │ WebSocket / REST Polling
             ▼
[Interactive Web Dashboard (Gauri)]
  ├── Live Camera Snap & Image Drag-and-Drop Uploader
  ├── Side-by-Side Original Leaf vs Grad-CAM Heatmap Viewer
  ├── Dynamic Plant Health Gauge & Risk Badges
  ├── Interactive 30-Day Vineyard Health & Humidity Trends
  └── English & Marathi (मराठी) Language Switcher
```

---

## 📐 Mathematical Formulation

### 1. Vapor Pressure Deficit (VPD in kPa)
Determines vine transpiration stress:
$$VP_{\text{sat}} = 0.61078 \times \exp\left(\frac{17.27 \times T}{T + 237.3}\right)$$
$$\text{VPD} = VP_{\text{sat}} \times \left(1 - \frac{\text{RH}}{100}\right)$$

### 2. Multi-Modal Fused Risk Modifier
Combines visual prediction probability $P_{\text{CV}}$ with environmental prior $R_{\text{env}}$:
$$P_{\text{fused}}(D_i) = \frac{P_{\text{CV}}(D_i) \cdot (1 + \alpha \cdot R_{\text{env}}(D_i))}{\sum_j P_{\text{CV}}(D_j) \cdot (1 + \alpha \cdot R_{\text{env}}(D_j))}$$

### 3. Composite Plant Health Score ($0\text{–}100$)
$$\text{Health Score} = 100 - \left( P_{\text{fused}}(D) \times 65 \times S_k + \text{NecrosisRatio} \times 0.35 + \text{VPD}_{\text{penalty}} \right)$$
* $S_k$: Disease severity coefficient (Downy Mildew: $1.0$, Powdery Mildew: $0.85$, Bacterial Canker: $0.80$, Healthy: $0.0$).
* $\text{NecrosisRatio}$: Surface lesion area quantified via OpenCV HSV thresholding.

---

## 🎯 10-Slide Project Presentation Pitch Deck Outline

### Slide 1: Title & Team
* **Title:** LeafLens AI Pro: AI-Based Grapevine Disease Detection & Smart Health Monitoring
* **Subtitle:** An IoT + Deep Learning Viticulture Decision Support System
* **Team Members:** Jiya Gavali, Rachi Gedam, Gargi Shinde, Gauri Patil
* **Institution:** Engineering College / University

### Slide 2: The Problem Statement (Viticulture in Nashik)
* Nashik is India's Grape Capital ($>70\%$ of exports).
* Fungal diseases (Downy Mildew, Powdery Mildew) and Bacterial Canker cause $40\%\text{–}80\%$ yield loss.
* Farmers over-spray expensive chemical fungicides preemptively due to lack of early detection, causing chemical resistance and pesticide residue rejections.

### Slide 3: Why Pure Computer Vision Fails in Vineyards
* Outdoor sunlight glares, shadows, and angle changes cause erratic classifications.
* Early-stage fungal spores visually mimic nutrient deficiencies.
* **Our Solution:** A controlled-illumination scanning chamber + microclimate sensor fusion.

### Slide 4: System Architecture & Workflow
* Hardware Scanning Chamber $\rightarrow$ Edge Capture Script $\rightarrow$ FastAPI Gateway $\rightarrow$ AI Inference & Grad-CAM $\rightarrow$ Farmer Dashboard.

### Slide 5: Hardware & Chamber Innovation (Rachi Gedam)
* Matte-black light-tight box with $45^\circ$ CRI 95+ diffused LED illumination.
* Sensirion SHT31 ($T/\text{RH}$), BH1750 (Lux), and Raspberry Pi Camera Module 3.
* Physical tactile button for one-touch field scanning.

### Slide 6: Deep Learning & Explainability (Jiya Gavali)
* Trained on 2,726 authentic leaves from Niphad, Nashik.
* Transfer learning using **MobileNetV3-Large** (lightweight, $<25\text{ MB}$).
* **Grad-CAM visual attention heatmaps** show farmers *why* the AI made the diagnosis.
* Achieved **$98.53\%$ validation accuracy**.

### Slide 7: Backend & Multi-Modal Decision Engine (Gargi Shinde)
* FastAPI asynchronous microservice architecture.
* Multi-Modal Risk Fusion combining visual probability with biological humidity/temp rules.
* Persistent SQLite database tracking vineyard scan history.

### Slide 8: Farmer-Centric Dashboard (Gauri Patil)
* Dual-language toggle: English & Marathi (मराठी).
* Live camera capture directly from mobile or laptop webcam.
* 30-day vineyard health trend charts for tracking disease progression.

### Slide 9: Agronomic Advisory (ICAR-NRCG Aligned)
* Precise fungicide schedules and chemical formulations (Metalaxyl + Mancozeb, Wettable Sulfur, Copper Oxychloride).
* Cultural practices: de-suckering, canopy aeration, pruning disinfection.

### Slide 10: Future Roadmap & Industry Impact
* Field trials with FPOs (Sahyadri Farms) and research centers (ICAR-NRCG Pune).
* Edge deployment directly on Raspberry Pi Zero 2W.
* Multi-spectral canopy drone integration.

---

## 🎬 3-Minute Live Demonstration Script for Evaluators

1. **Minute 1: The Problem & Hardware Showcase**
   * *"Respected evaluators, in viticulture, early detection is critical. Here is our LeafLens scanning chamber design by Rachi, featuring a controlled LED illumination ring and temperature/humidity sensors to eliminate outdoor lighting artifacts."*
2. **Minute 2: Live Diagnostic Scan & Explainability**
   * *Open the dashboard at `http://localhost:8501`.*
   * *"Let us select a Downy Mildew leaf from our Niphad test library (or snap a live photo). As you can see, the system predicts Downy Mildew with 100% confidence. Notice the Grad-CAM heatmap: the AI highlights the exact yellow-brown oil spots on the leaf."*
3. **Minute 3: Multi-Modal Fusion, History & Marathi Advisory**
   * *"Watch what happens when I adjust the Humidity slider above 80%: the multi-modal fusion engine flags a 'CRITICAL' sporulation alert. When we switch the language to Marathi, local farmers immediately see: 'डाऊनी मिल्ड्यू (केवडा)' with exact ICAR-NRCG spray formulations. Scroll down to see our 30-day temporal trend chart tracking vineyard health over time."*
