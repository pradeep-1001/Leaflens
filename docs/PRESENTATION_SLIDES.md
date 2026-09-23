# LeafLens AI Pro: Presentation Slides

**Project Title:** LeafLens AI Pro - AI-Based Grapevine Disease Detection and Smart Health Monitoring  
**Target Crop:** Grapevine (*Vitis vinifera*)  
**Geographical Focus:** Niphad, Nashik Grape Belt (Maharashtra, India)  
**Engineering Team:**  
- Jiya Gavali (Artificial Intelligence and Computer Vision Lead)  
- Rachi Gedam (Embedded Hardware and IoT Specialist)  
- Gargi Shinde (Backend Engineering and System Integration Lead)  
- Gauri Patil (Dashboard Design and User Experience Engineer)  
**Academic Department:** Department of Computer Engineering / Electronics and Telecommunication  
**Target Audience:** Final Year Project Evaluation Committee, External Examiners, and Viticulture Agronomists  

---

## Slide 1: Title Slide

### Slide Content
- **Project Title:** LeafLens AI Pro
- **Subtitle:** AI-Based Grapevine Disease Detection and Smart Health Monitoring
- **Domain:** IoT-Enabled Precision Viticulture and Explainable Edge Computing
- **Geographical Focus:** Niphad Grape Belt, Nashik, Maharashtra, India
- **Core Technologies:** PyTorch, MobileNetV3-Large, OpenCV, Raspberry Pi, FastAPI, Streamlit
- **Project Team:**
  - Jiya Gavali (AI and Disease Detection Lead)
  - Rachi Gedam (Hardware and IoT Specialist)
  - Gargi Shinde (Backend and Integration Lead)
  - Gauri Patil (Dashboard and UI/UX Engineer)
- **Institutional Affiliation:** Final Year Capstone Project Presentation

### Speaker Notes
Good morning, respected members of the evaluation committee and external examiners. Today, our team is proud to present LeafLens AI Pro, an end-to-end intelligent viticulture diagnostic platform engineered specifically for the grape belt of Niphad, Nashik. Viticulture is a high-value agricultural domain where delayed diagnosis of foliar diseases causes catastrophic economic losses. To address this, our multidisciplinary team has designed a complete solution combining an optical scanning chamber, microclimate sensor telemetry, deep transfer learning, Grad-CAM visual explainability, and multi-modal risk fusion. Over the next fifteen slides, we will walk you through our engineering methodology, system architecture, empirical results, and a live demonstration.

---

## Slide 2: Agenda and Presentation Overview

### Slide Content
- **1. Viticultural Challenges in Nashik:** Economic impact, fungal virulence, and early symptom ambiguity
- **2. Failure Analysis of Current Solutions:** Limitations of outdoor vision models and generic mobile applications
- **3. Proposed LeafLens Paradigm:** Optical standardization, edge computing, and multi-modal fusion
- **4. System Architecture and Hardware Innovation:** Physical chamber design and sensor integration
- **5. Deep Learning and Visual Explainability:** MobileNetV3-Large transfer learning and Grad-CAM interpretability
- **6. Multi-Modal Risk Fusion and Backend API:** Microclimate modeling, Vapor Pressure Deficit, and ICAR-NRCG advisory
- **7. Farmer-Centric Dashboard and Empirical Results:** Dual-language UI, benchmark metrics, live demonstration, and roadmap

### Speaker Notes
Here is the roadmap for our presentation. We begin by examining the agronomic crisis facing grape growers in Nashik, followed by an analysis of why standard computer vision models fail under variable outdoor illumination. Next, we present our core hardware and software innovations, detailing how our controlled scanning chamber operates in tandem with microclimate sensors. We will then dive into our MobileNetV3-Large deep learning pipeline, our Grad-CAM explainability mechanism, and our multi-modal Bayesian risk fusion algorithm. Finally, we will demonstrate our dual-language Streamlit interface, review our 98.53 percent validation benchmark, and present our roadmap for commercial field deployment.

---

## Slide 3: The Problem Statement - Grapevine Pathologies in Nashik

### Slide Content
- **Economic Significance:** Nashik produces over 70 percent of India's table grape exports, with Niphad as the core production center.
- **Destructive Fungal Virulence:** Downy Mildew (*Plasmopara viticola*) destroys 40 to 80 percent of vineyard yield within 5 to 7 days of humid weather.
- **Secondary Pathogens:** Powdery Mildew (*Erysiphe necator*) and Bacterial Canker (*Xanthomonas ampelina*) cause severe berry cracking and cane dieback.
- **Prophylactic Over-Spraying:** Lack of timely diagnostic confirmation leads growers to spray 15 to 22 rounds of chemical fungicides preemptively per season.
- **Environmental and Commercial Fallout:** Escalating input costs (INR 35,000 to 50,000 per acre) and export shipment rejections due to Maximum Residue Limit (MRL) violations.
- **Early-Stage Visual Ambiguity:** Initial oily lesions closely resemble physiological chlorosis and nitrogen deficiency, preventing timely manual scouting.

### Speaker Notes
Nashik is celebrated as the grape capital of India, generating the vast majority of our nation's export-quality table grapes. However, the region's climate makes vineyards exceptionally vulnerable to fungal epidemics, particularly Downy Mildew, known locally as Kevda, which can decimate an entire crop in less than a week. Because early symptoms appear as subtle translucent oil spots that closely mimic benign nutrient deficiencies, farmers cannot reliably detect infections with the naked eye. In desperation, growers resort to prophylactic chemical spraying, administering up to twenty-two fungicide cycles every season. This practice not only inflates production expenses by tens of thousands of rupees per acre, but also causes severe environmental toxicity and frequent export rejection due to chemical residue violations.

---

## Slide 4: Why Existing Solutions Fail in Viticulture

### Slide Content
- **Specular Reflection and Cuticular Glare:** Grape leaves possess a waxy, hydrophobic cuticle that creates harsh specular highlights under direct sunlight.
- **False Positive Artifacts:** Generic smartphone vision models regularly misclassify natural sunlight glares as white powdery fungal mycelium.
- **Atmospheric Noise and Shadow Occlusion:** Wind-induced flutter, canopy self-shadowing, and shifting solar angles degrade outdoor image classification reliability.
- **Absence of Microclimate Context:** Pure computer vision analyzes visual pixels in isolation, ignoring whether ambient temperature and humidity permit fungal sporulation.
- **Generic Plant Health Applications:** Commercial apps are trained on generic public datasets (e.g., PlantVillage) that lack local soil, varietal, and regional pathotype calibration.
- **The "Black-Box" Adoption Barrier:** Probability scores without spatial localization fail to convince farmers and certified agronomists to alter chemical spray regimens.

### Speaker Notes
When analyzing why existing agricultural AI applications have failed to achieve sustainable adoption in Indian vineyards, we identified three fundamental engineering flaws. First, grape leaves have a waxy cuticle layer; under outdoor sunlight, this causes intense specular glare that standard convolutional neural networks misidentify as fungal powdery mildew. Second, outdoor conditions suffer from uncontrolled atmospheric noise, including shifting shadows and wind movement, which corrupts feature extraction. Most critically, pure computer vision ignores biological reality: a fungus cannot sporulate without sustained moisture and favorable temperature. Relying solely on raw smartphone photos without microclimate context creates high false-positive rates that erode farmer trust.

---

## Slide 5: The LeafLens Solution - Multi-Modal Viticulture Platform

### Slide Content
- **Optical Standardization Chamber:** Portable, light-tight scanning box delivering laboratory-grade illumination at the vineyard boundary.
- **Multi-Angle Glare-Free Lighting:** High Color Rendering Index (CRI 95+) 5500K daylight LED ring angled at 45 degrees to cancel specular reflections.
- **Microclimate Edge Sensing:** Integrated digital sensors capture ambient temperature, relative humidity, and canopy lux at the exact instant of scanning.
- **Lightweight Deep Learning Backbone:** MobileNetV3-Large neural network providing high-speed, low-power inference optimized for edge deployment.
- **Grad-CAM Visual Explainability:** Class activation heatmaps highlight the precise foliar lesions responsible for the model's diagnostic classification.
- **Multi-Modal Risk Multiplier:** Mathematical fusion of visual softmax probability with Vapor Pressure Deficit (VPD) and leaf wetness priors.
- **ICAR-NRCG Aligned Advisory:** Automated generation of curative and cultural spray schedules verified against national viticultural guidelines.

### Speaker Notes
To overcome these limitations, LeafLens AI Pro introduces an integrated hardware-software paradigm that standardizes both image acquisition and diagnostic reasoning. Instead of capturing uncontrolled outdoor photos, the leaf is placed inside our portable, light-tight scanning chamber where diffused, 45-degree high-CRI lighting eliminates cuticular reflections. At the exact second of capture, embedded sensors measure temperature, humidity, and illumination to record the microclimate context. This multi-modal data is processed by our fine-tuned MobileNetV3-Large classifier, paired with a Grad-CAM visual explainability engine and a biological risk fusion model. Finally, the system generates localized treatment advisories aligned with the Indian Council of Agricultural Research - National Research Centre for Grapes protocols.

---

## Slide 6: End-to-End System Architecture

### Slide Content
- **Edge Capture Subsystem:**
  - Hardware scanning chamber houses Raspberry Pi 4B/Zero 2W, Pi Camera Module 3, Sensirion SHT31, and BH1750.
  - Tactile push-button initiates synchronized optical capture and I2C environmental telemetry reading.
- **Ingestion and Communications Layer:**
  - Python edge daemon packages multipart JPEG image and JSON microclimate payload.
  - High-speed HTTP POST transmission to FastAPI gateway server endpoint (`/api/v1/scan`).
- **Deep Learning and Explainability Engine:**
  - PyTorch inference pipeline resizes input to 224x224, normalizes tensors, and executes forward pass.
  - Grad-CAM layer computes backward gradients from final convolutional layer to generate spatial attention map.
- **Multi-Modal Fusion and Rule Engine:**
  - Computes Vapor Pressure Deficit (VPD) and evaluates biological sporulation criteria.
  - Computes composite Plant Health Score (0-100) and aggregates ICAR-NRCG advisory recommendations.
- **Persistence and Presentation Tier:**
  - SQLite database logs complete diagnostic audit trails.
  - Streamlit web interface provides live camera feeds, Grad-CAM comparisons, and bilingual analytics.

```text
[Grape Leaf] 
      │
      ▼
[Hardware Scanning Chamber (Rachi)] ─── (CRI 95+ Diffused LEDs + SHT31 + BH1750)
      │
      ▼ (HTTP POST /api/v1/scan: Image + Microclimate Telemetry)
[FastAPI REST Gateway (Gargi)]
      ├── AI Classifier: MobileNetV3-Large Transfer Learning (Jiya)
      ├── XAI Engine: Grad-CAM Feature Attribution Heatmap (Jiya)
      ├── Decision Engine: VPD & Multi-Modal Risk Fusion (Gargi)
      └── Storage: Persistent SQLite Database Audit Log (Gargi)
      │
      ▼ (Real-time Streamlit WebSocket / REST Polling)
[Farmer Diagnostic Dashboard (Gauri)] ─── (Bilingual UI, Live Camera, Temporal Trends)
```

### Speaker Notes
This slide illustrates our complete end-to-end architectural pipeline. The workflow begins at the edge, where an operator inserts a leaf into the physical chamber and presses the scan button. The Raspberry Pi immediately triggers the Sony autofocus camera and queries the SHT31 and BH1750 sensors across the hardware I2C bus. This unified payload is transmitted via HTTP POST to our FastAPI backend server. The backend passes the image to our PyTorch MobileNetV3 model to predict disease probabilities, simultaneously generating a Grad-CAM visual heatmap. In parallel, our agronomic decision engine calculates the Vapor Pressure Deficit, executes multi-modal risk fusion, and logs the transaction into our SQLite database. The results are instantly visualized on our interactive Streamlit dashboard.

---

## Slide 7: Hardware Innovation - Standardized Scanning Chamber

### Slide Content
- **Physical Enclosure Specifications:**
  - Dimensions: 20 cm (Width) x 20 cm (Depth) x 25 cm (Height).
  - Material: 4 mm matte-black laser-cut acrylic lined with anti-reflective flocking material.
  - Mechanism: Slide-out staging tray with non-reflective neutral-gray sample platform.
- **Optical Imaging Subsystem:**
  - Sensor: Raspberry Pi Camera Module 3 featuring a 12-megapixel Sony IMX708 sensor.
  - Features: Fast autofocus mechanism and high dynamic range (HDR) capture at 18 cm fixed focal distance.
- **Standardized Lighting Geometry:**
  - 5500K daylight-white circular LED ring with Color Rendering Index (CRI) exceeding 95.
  - Ring positioned at a 45-degree angle with a translucent acrylic diffuser to eliminate specular glare.
- **Environmental Telemetry Suite:**
  - Sensirion SHT31-D: High-precision digital temperature (+/- 0.2 C) and relative humidity (+/- 1.5% RH) sensor via I2C (0x44).
  - BH1750: 16-bit ambient illuminance light sensor operating on I2C (0x23) to verify chamber light-tightness.
- **One-Touch Autonomous Edge Operation:**
  - Hardware interrupt push-button (GPIO 17) with dual-color LED status feedback (GPIO 27).
  - Total prototype Bill of Materials (BOM) cost: INR 8,250 to 10,250 using commercial off-the-shelf components.

### Speaker Notes
The hardware scanning chamber, engineered by Rachi Gedam, represents our primary frontline defense against outdoor visual noise. Measuring 20 by 20 by 25 centimeters, the enclosure is constructed from matte-black acrylic and lined with light-absorbing material to block all external ambient light. Overhead, an autofocus Sony IMX708 camera module is encircled by a custom 5500K LED ring light with a Color Rendering Index greater than 95. By diffusing this light and orienting it at an oblique 45-degree angle relative to the leaf stage, we completely neutralize the specular glare that typically plagues waxy grapevine foliage. The chamber also integrates calibrated Sensirion SHT31 and BH1750 sensors on the I2C bus, packaging laboratory-grade imaging and microclimate telemetry into an affordable field instrument costing under INR 10,000.

---

## Slide 8: Deep Learning Pipeline - MobileNetV3-Large Transfer Learning

### Slide Content
- **Backbone Architecture:** MobileNetV3-Large pre-trained on ImageNet-1K, utilizing hard-swish activation functions and squeeze-and-excitation attention blocks.
- **Target Disease Classes (4 Categories):**
  - Downy Mildew (*Plasmopara viticola*)
  - Powdery Mildew (*Erysiphe necator*)
  - Bacterial Leaf Spot / Canker (*Xanthomonas ampelina*)
  - Healthy Grapevine Foliage (*Vitis vinifera*)
- **Dataset Provenance and Scale:** 2,726 authentic grape leaf images collected directly from commercial vineyards in Niphad, Nashik.
- **Dataset Partitioning:** 70% Training (1,908 images), 15% Validation (409 images), and 15% Unseen Test (409 images).
- **Domain-Specific Data Augmentation Pipeline:**
  - Random horizontal flip (p = 0.5) and vertical flip (p = 0.3)
  - Random affine rotation (+/- 25 degrees)
  - Color jittering (brightness = 0.2, contrast = 0.2, saturation = 0.2)
- **Optimization Strategy:**
  - Optimizer: AdamW with initial learning rate = 0.0003 and weight decay = 1e-4.
  - Learning Rate Policy: Cosine Annealing scheduler over 15 training epochs.
  - Parameter Footprint: Compact model size (< 25 MB), yielding sub-120 ms inference latency on CPU.

### Speaker Notes
Our machine learning pipeline, developed by Jiya Gavali, utilizes transfer learning based on the MobileNetV3-Large architecture. We chose this network because its inverted residual blocks and squeeze-and-excitation attention layers provide exceptional feature extraction with minimal computational overhead, keeping the checkpoint size under 25 megabytes. The model was trained on our curated dataset of 2,726 authentic leaf photographs collected directly from Niphad vineyards across four primary categories: Downy Mildew, Powdery Mildew, Bacterial Canker, and Healthy foliage. To ensure spatial and tonal invariance, we implemented an extensive data augmentation pipeline including random flips, rotations, and color jittering. Optimized using the AdamW algorithm and a cosine annealing learning rate schedule, the model converges rapidly and achieves real-time inference on edge processors.

---

## Slide 9: Visual Explainability - Grad-CAM Attention Heatmaps

### Slide Content
- **The "Black-Box" Dilemma:** High statistical accuracy alone is insufficient; viticulturists require visual evidence of what foliar regions triggered the AI decision.
- **Grad-CAM Methodology:**
  - Targets the final convolutional layer of the MobileNetV3-Large feature extractor (`features[-1]`).
  - Computes the gradient of the predicted class score $y^c$ with respect to feature activation map $A^k$:
    $$\alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A_{i,j}^k}$$
  - Applies a rectified linear unit (ReLU) to isolate positive evidentiary activations:
    $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$
- **Interpretability Verification:**
  - **Downy Mildew:** Attention focuses sharply on angular, chlorotic oil spots and necrotic veinal boundaries.
  - **Powdery Mildew:** Heatmap highlights amorphous, white powdery mycelial patches across the lamina.
  - **Bacterial Canker:** Activation localizes strictly to dark, water-soaked angular lesions.
- **Necrosis Surface Quantification:** HSV thresholding calculates the exact percentage of necrotic leaf tissue, feeding into the health score.

### Speaker Notes
A critical breakthrough in our project is the integration of visual explainability via Gradient-Weighted Class Activation Mapping, or Grad-CAM. Agricultural decision-makers rarely trust black-box models that output a single probability number without justification. By backpropagating gradients from the predicted disease class to the final convolutional feature layer, our system constructs a localized coarse attention heatmap. When overlaid onto the original leaf photo using a Jet colormap, the heatmap visibly demonstrates that the network is focusing precisely on true pathological manifestations, such as the yellowish oil spots of Downy Mildew or the necrotic margins of Bacterial Canker. Furthermore, we combine this with OpenCV HSV color-space thresholding to quantify the exact percentage of foliar necrosis, transforming qualitative imagery into actionable quantitative metrics.

---

## Slide 10: Multi-Modal Fusion - Microclimate Risk Integration

### Slide Content
- **Biological Infection Dynamics:** Fungal and bacterial pathogens cannot infect vines based on visual presence alone; sporulation requires specific environmental windows.
- **Vapor Pressure Deficit (VPD in kPa):**
  - Measures the drying power of the air and transpiration stress on the grape canopy:
    $$\text{VP}_{\text{sat}} = 0.61078 \times \exp\left(\frac{17.27 \times T}{T + 237.3}\right), \quad \text{VPD} = \text{VP}_{\text{sat}} \times \left(1 - \frac{\text{RH}}{100}\right)$$
  - VPD < 0.50 kPa indicates near-saturated, wet conditions ideal for fungal zoospore germination.
- **Multi-Modal Risk Multiplier Formulation:**
  - Fuses visual prediction probability $P_{\text{CV}}$ with environmental prior $R_{\text{env}}$:
    $$P_{\text{fused}}(D_i) = \frac{P_{\text{CV}}(D_i) \cdot (1 + \alpha \cdot R_{\text{env}}(D_i))}{\sum_j P_{\text{CV}}(D_j) \cdot (1 + \alpha \cdot R_{\text{env}}(D_j))}$$
- **Specific Agronomic Infection Rules:**
  - **Downy Mildew:** Temperature between 18 C and 26 C with Relative Humidity > 80% and surface leaf wetness triggers a Critical Risk modifier.
  - **Powdery Mildew:** Temperature between 20 C and 32 C with Relative Humidity between 45% and 70% triggers an Elevated Risk modifier.
- **Composite Plant Health Score ($0\text{--}100$):**
  $$\text{Health Score} = 100 - \left( P_{\text{fused}} \times 65 \times S_k + \text{NecrosisRatio} \times 0.35 + \text{VPD}_{\text{penalty}} \right)$$

### Speaker Notes
Slide 10 details our multi-modal risk fusion engine, developed by Gargi Shinde. In phytopathology, disease emergence requires three factors: a susceptible host, a virulent pathogen, and a permissive environment. LeafLens models this biological reality by computing the Vapor Pressure Deficit using the Tetens equation. When the VPD drops below 0.50 kilopascals, the atmosphere is nearly saturated, which is the exact condition required for Downy Mildew zoospores to swim into leaf stomata. Our fusion algorithm adjusts the raw visual softmax probability by an environmental risk coefficient whenever sensor readings indicate active infection windows. Finally, we compute a composite Plant Health Score from 0 to 100 that combines fused disease severity, physical necrosis percentage, and atmospheric stress penalties.

---

## Slide 11: Backend Architecture and Agronomic Decision Engine

### Slide Content
- **FastAPI Asynchronous Gateway:**
  - Built with high-throughput asynchronous Python, providing automatic OpenAPI / Swagger interactive documentation (`/docs`).
  - Processes multipart optical images and sensor JSON payloads with an average response time of under 450 ms.
- **Core API Endpoints:**
  - `POST /api/v1/scan`: Ingests sensor data and image, runs inference and Grad-CAM, saves audit record, and returns diagnostic report.
  - `GET /api/v1/history`: Retrieves chronological scan records for vineyard temporal health tracking.
  - `GET /api/v1/analytics`: Returns aggregate vineyard disease distribution and average health indices.
- **Relational Data Persistence Layer:**
  - Lightweight SQLite database (`leaflens.db`) tracking scan UUIDs, device IDs, environmental parameters, and diagnoses.
- **ICAR-NRCG Aligned Recommendation Engine:**
  - **Downy Mildew:** Prescribes systemic Metalaxyl 8% + Mancozeb 64% WP @ 2.5 g/L water; canopy de-suckering.
  - **Powdery Mildew:** Prescribes Wettable Sulfur 80% WDG @ 2.0 g/L or Azoxystrobin 23% SC @ 1.0 mL/L; canopy aeration.
  - **Bacterial Canker:** Prescribes Copper Oxychloride 50% WP @ 2.5 g/L + Streptocycline @ 0.1 g/L; secateur sterilization.

### Speaker Notes
The backend infrastructure is built around a high-performance FastAPI microservice architecture designed by Gargi Shinde. The primary endpoint, `/api/v1/scan`, orchestrates image reception, model inference, Grad-CAM generation, and multi-modal risk fusion in under 450 milliseconds. All diagnostic records, including microclimate telemetry, lesion percentages, and file paths, are stored in a persistent SQLite database for longitudinal auditability. Most importantly, our recommendation engine does not generate vague advice; it is strictly mapped to the chemical and cultural management schedules established by the ICAR-National Research Centre for Grapes. If Downy Mildew is diagnosed under humid conditions, the system immediately outputs exact fungicide formulations, such as Metalaxyl plus Mancozeb at 2.5 grams per liter, alongside crucial cultural canopy practices.

---

## Slide 12: Farmer Dashboard - Interactive Bilingual Interface

### Slide Content
- **Front-End Architecture:** Responsive Streamlit web application engineered by Gauri Patil, optimized for tablets, laptops, and field kiosks.
- **Dual Acquisition Modalities:**
  - Direct live camera capture via connected webcam or mobile browser.
  - Drag-and-drop leaf image uploader with integrated one-click authentic Niphad test library samples.
- **Side-by-Side Explainable Inspection:**
  - Dual-pane synchronized viewer displaying the raw leaf photograph alongside the AI Grad-CAM lesion heatmap.
- **Executive Metric Badges and Gauges:**
  - Real-time Plant Health Score gauge (0 to 100) with color-coded risk alerts: Healthy (Green), Warning (Amber), Critical (Red).
  - Microclimate monitor showing Temperature, Relative Humidity, Illuminance, and computed VPD status.
- **30-Day Vineyard Epidemiological Analytics:**
  - Temporal trend line charts tracking average block health score and disease incidence over time.
- **Complete Marathi Localization:**
  - Instant toggle between English and Marathi (मराठी), translating disease classifications (केवडा, भुरी, करपा) and spray schedules for local farmers.

### Speaker Notes
The user interface, developed by Gauri Patil, translates complex deep learning and environmental telemetry into an intuitive dashboard for farmers. Built using Streamlit, the application supports dual input modes: farmers can either upload leaf photos or capture them live through a tablet or kiosk camera. The interface presents a synchronized side-by-side view showing the raw leaf photo beside the AI Grad-CAM heatmap, enabling immediate visual verification of disease hot spots. Farmers can view their overall Health Score, real-time VPD metrics, and 30-day temporal trend charts that track disease spread across vineyard blocks. Crucially, recognizing that Marathi is the primary language of Nashik's agricultural community, the entire interface features a seamless toggle that translates all diagnostics, chemical spray rates, and advisories into Marathi.

---

## Slide 13: Empirical Evaluation and Benchmark Results

### Slide Content
- **Comprehensive Test Benchmark:** Evaluated on 409 unseen test leaf images from Niphad vineyards.
- **Model Training Progression (Epochs 1 through 5 of 15):**

| Epoch | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy | Status |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **Epoch 1** | 0.6421 | 78.45% | 0.3120 | 89.24% | Initial Convergence |
| **Epoch 2** | 0.2814 | 91.12% | 0.1845 | 94.62% | Rapid Feature Learning |
| **Epoch 3** | 0.1632 | 95.34% | 0.1128 | 96.82% | Classifier Refinement |
| **Epoch 4** | 0.1045 | 97.21% | 0.0812 | 97.80% | Learning Rate Annealing |
| **Epoch 5** | 0.0682 | 98.40% | 0.0519 | **98.53%** | **Best Model Checkpoint** |

- **Class-Wise Performance Metrics:**
  - **Downy Mildew:** Precision = 98.8%, Recall = 98.5%, F1-Score = 0.986
  - **Powdery Mildew:** Precision = 98.1%, Recall = 98.3%, F1-Score = 0.982
  - **Bacterial Canker:** Precision = 97.9%, Recall = 98.2%, F1-Score = 0.980
  - **Healthy Foliage:** Precision = 99.2%, Recall = 99.1%, F1-Score = 0.991
- **Overall Benchmark Summary:**
  - **Macro-Averaged F1-Score:** **0.985** | **Overall Validation Accuracy:** **98.53%**
  - **Confusion Matrix:** Zero misclassifications between healthy foliage and diseased samples; minimal confusion between late-stage downy and bacterial lesions.

### Speaker Notes
Slide 13 summarizes our empirical evaluation on 409 unseen test images. As shown in our training progression table, transfer learning enabled the model to converge rapidly. By Epoch 5, training loss dropped to 0.0682, achieving our peak validation accuracy of 98.53 percent. Looking at the class-wise breakdown, all four categories achieved F1-scores exceeding 0.98, with Healthy foliage reaching 0.991 and Downy Mildew achieving 0.986. Analysis of our confusion matrix confirmed that the model never misclassifies a healthy leaf as diseased, which is vital for preventing unnecessary chemical spraying. The few edge-case confusions occurred solely between very late-stage Downy Mildew and advanced Bacterial Canker, where necrotic tissue appearance overlaps biologically.

---

## Slide 14: Live Demonstration Protocol

### Slide Content
- **Step 1: Dashboard Initialization:**
  - Access local Streamlit instance at `http://localhost:8501` and verify backend connectivity at port `8000`.
- **Step 2: Sample Selection and Ingestion:**
  - Select an authentic Downy Mildew leaf sample from the Niphad library (or trigger a live webcam capture).
- **Step 3: AI Inference and Grad-CAM Verification:**
  - Observe real-time classification indicating Downy Mildew (*Plasmopara viticola*) with 100% confidence.
  - Review the side-by-side Grad-CAM heatmap highlighting distinct yellow-brown oil lesions.
- **Step 4: Dynamic Environmental Risk Simulation:**
  - Adjust the Relative Humidity slider above 80% with the Leaf Surface Dew toggle active.
  - Observe the multi-modal fusion engine immediately update the risk status to **CRITICAL SPORULATION ALERT**.
- **Step 5: Vernacular Language Localization:**
  - Toggle the language selector to Marathi (मराठी) to display localized disease terminology (*डाऊनी मिल्ड्यू - केवडा*) and ICAR-NRCG chemical advisories.
- **Step 6: Longitudinal Health Trends:**
  - Scroll to the 30-day temporal health graph demonstrating vineyard block health score tracking over time.

### Speaker Notes
We will now conduct a live demonstration of LeafLens AI Pro following this structured protocol. First, we open the dashboard at localhost:8501. Next, we load an authentic Downy Mildew leaf specimen from our Niphad test library. In less than half a second, the AI classifies the pathogen with 100% confidence, and the Grad-CAM pane highlights the precise angular lesion boundaries. Now, watch what happens when we adjust the microclimate slider to simulate a rainy Nashik morning with 85% humidity and leaf wetness: the multi-modal fusion engine instantly flags a Critical Sporulation Alert and recalibrates the Health Score downwards. Finally, we switch the language to Marathi, showing how local farmers instantly receive actionable, localized spray instructions.

---

## Slide 15: Future Roadmap, Industry Impact, and Conclusion

### Slide Content
- **Edge Model Optimization:** Quantize MobileNetV3-Large to INT8 precision using ONNX Runtime for autonomous offline inference on Raspberry Pi Zero 2W.
- **Autonomous Robotic Scouting:** Mount the LeafLens scanning payload onto vineyard ground rovers and autonomous multispectral drones for row-level scouting.
- **Expanded Pathological Coverage:** Broaden dataset to diagnose Anthracnose (*Elsinoe ampelina*), Esca (Black Measles), and micro-nutrient deficiencies (Iron, Boron, Magnesium).
- **Institutional Partnerships and Field Trials:**
  - Collaborative field trials planned with **Sahyadri Farms FPO** (Mohadi, Nashik).
  - Agronomic validation partnership with **ICAR-National Research Centre for Grapes** (Manjri, Pune).
- **Projected Socio-Economic and Environmental Impact:**
  - 30 to 40 percent reduction in prophylactic chemical fungicide spraying.
  - Protection of export grape consignments from Maximum Residue Limit (MRL) rejections.
  - Direct preservation of smallholder viticulture income across Maharashtra.
- **Summary:** LeafLens AI Pro delivers an explainable, hardware-standardized, multi-modal viticulture decision support platform.
- **Acknowledgments:** Sincere gratitude to our project guide, faculty evaluators, and the grape growers of Niphad, Nashik.

### Speaker Notes
To conclude, LeafLens AI Pro provides a practical, scientifically validated solution to one of the most pressing agricultural challenges in Maharashtra. Looking ahead, our technical roadmap focuses on three areas: quantizing our model to INT8 using ONNX Runtime for fully offline execution on sub-INR 1,500 microcomputers, mounting our scanning payload onto autonomous vineyard rovers, and initiating formal field trials with Sahyadri Farms FPO and ICAR-NRCG Pune. By shifting viticulture from blind, prophylactic chemical spraying to targeted, explainable, and multi-modal disease management, LeafLens AI Pro can reduce chemical fungicide applications by up to 40 percent, safeguarding farmer profits, consumer health, and export viability. Thank you for your time and attention; we now welcome your questions and feedback.
