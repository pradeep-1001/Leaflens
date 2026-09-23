# LeafLens AI Pro — AI-Based Grapevine Disease Detection & Smart Health Monitoring

**PROJECT REPORT**

**Team Members:**
- Jiya Gavali (AI & Vision)
- Rachi Gedam (Hardware & IoT)
- Gargi Shinde (Backend & Integration)
- Gauri Patil (Dashboard & UI)

**Institution:** [Insert Institution Name]
**Date:** September 2026

---

# CERTIFICATE

This is to certify that the project report entitled **"LeafLens AI Pro — AI-Based Grapevine Disease Detection & Smart Health Monitoring"** submitted by Jiya Gavali, Rachi Gedam, Gargi Shinde, and Gauri Patil is a bonafide work carried out by them under my supervision and guidance. This report is submitted towards partial fulfillment of the requirements for the award of degree of Bachelor of Engineering/Technology in Computer Engineering.

**Signature of Guide:** _________________
**Name of Guide:** ____________________
**Date:** _________________

---

# ACKNOWLEDGEMENT

We would like to express our deepest gratitude to all those who provided us the possibility to complete this project. A special thanks to our project guide for their continuous support, patience, and expert guidance throughout the development of LeafLens AI Pro.

We also thank our institution for providing the necessary infrastructure and resources. Finally, we express our sincere appreciation to the viticulture experts and farmers in the Niphad, Nashik Grape Belt, whose insights and cooperation were invaluable in gathering the dataset and understanding the real-world challenges of grapevine disease management.

---

# ABSTRACT

The Nashik region in Maharashtra, particularly the Niphad belt, is a critical hub for grape cultivation in India. However, the grapevine industry suffers significant economic losses due to diseases such as Downy Mildew, Powdery Mildew, and Bacterial Leaf Spot. Traditional disease identification relies on manual inspection, which is labor-intensive, error-prone, and often delayed. This project introduces **LeafLens AI Pro**, a comprehensive, multi-modal system for early, accurate, and automated grapevine disease detection and health monitoring. The system integrates advanced computer vision, edge IoT, and agronomic analytics. Utilizing a MobileNetV3-Large transfer learning architecture fine-tuned on a custom dataset of 2,726 authentic images from Niphad vineyards, the AI model achieves a robust 98.53% validation accuracy across four classes. Crucially, the solution goes beyond simple image classification by incorporating a physical scanning chamber with controlled lighting, and fusing visual data with real-time environmental metrics (Temperature and Humidity via IoT sensors). This sensor fusion calculates Vapor Pressure Deficit (VPD) and environmental risk indices to modulate AI predictions, yielding highly reliable health scores. Furthermore, Grad-CAM explainability builds trust with end-users by visualizing the AI's focal points. The system features a FastAPI backend, an SQLite database, and a bilingual (English/Marathi) Streamlit dashboard that provides actionable, ICAR-NRCG aligned spray recommendations, thereby empowering farmers with precision agriculture tools to mitigate yield loss and optimize chemical usage.

---

# TABLE OF CONTENTS

1. [CHAPTER 1: INTRODUCTION](#chapter-1-introduction)
2. [CHAPTER 2: LITERATURE REVIEW](#chapter-2-literature-review)
3. [CHAPTER 3: SYSTEM DESIGN & ARCHITECTURE](#chapter-3-system-design--architecture)
4. [CHAPTER 4: IMPLEMENTATION](#chapter-4-implementation)
5. [CHAPTER 5: RESULTS & DISCUSSION](#chapter-5-results--discussion)
6. [CHAPTER 6: CONCLUSION & FUTURE WORK](#chapter-6-conclusion--future-work)
7. [REFERENCES](#references)
8. [APPENDIX A: Hardware Bill of Materials](#appendix-a-hardware-bill-of-materials)
9. [APPENDIX B: API Endpoint Documentation](#appendix-b-api-endpoint-documentation)
10. [APPENDIX C: Full Source Code Listings](#appendix-c-full-source-code-listings)

---

# CHAPTER 1: INTRODUCTION

## 1.1 Background & Motivation
Grape cultivation is a cornerstone of the agricultural economy in Maharashtra, India. The Nashik region, often dubbed the "Wine Capital of India," produces a significant proportion of the country's grapes, with the Niphad belt being particularly prominent. The livelihood of thousands of farmers and laborers depends on the successful yield of these vineyards. However, *Vitis vinifera* is highly susceptible to various fungal and bacterial infections. Diseases such as Downy Mildew, Powdery Mildew, and Bacterial Leaf Spot can devastate entire vineyards within a matter of days if left unchecked, especially under favorable humid conditions. The economic impact of these diseases is twofold: direct loss of yield and quality, and the exorbitant costs associated with indiscriminate and preemptive fungicide application, which also poses severe environmental and health hazards. There is a pressing need for a modernized, technological intervention that can accurately and promptly identify disease onset to facilitate targeted treatment.

## 1.2 Problem Statement
While general plant disease classification models exist (often trained on idealized laboratory datasets like PlantVillage), standard image classifiers fail catastrophically when deployed in real-world vineyards. The failures stem from uncontrolled lighting, varied backgrounds, diverse symptom presentations at different growth stages, and the lack of contextual environmental data. A farmer photographing a leaf under harsh sunlight or deep shade often receives incorrect diagnoses from standard apps. Furthermore, disease manifestation is strongly coupled with micro-climatic conditions; a visual symptom alone might not convey the full risk profile if environmental factors (like high humidity) are actively exacerbating the pathogen's spread. A robust solution must standardize the capture environment and fuse visual AI with agronomic context.

## 1.3 Objectives of the Project
The primary objectives of the LeafLens AI Pro project are:
1.  To develop a highly accurate, lightweight deep learning model capable of classifying primary grapevine diseases prevalent in the Nashik region.
2.  To design and build an IoT-enabled controlled environment scanning chamber to eliminate lighting and background variability during image capture.
3.  To integrate real-time environmental sensors (Temperature, Humidity, Light) to capture micro-climatic data concurrently with leaf images.
4.  To implement a multi-modal data fusion algorithm that combines AI visual confidence with environmental risk indices (e.g., VPD).
5.  To provide clear, explainable AI outputs (using Grad-CAM) and localized, actionable recommendations aligned with ICAR-NRCG guidelines through a user-friendly, bilingual interface.

## 1.4 Scope & Limitations
**Scope:** The project is limited to *Vitis vinifera* (grapevine) leaves and specifically targets three major diseases: Downy Mildew, Powdery Mildew, and Bacterial Leaf Spot, alongside a Healthy class. The geographical focus for dataset collection and validation is the Niphad, Nashik Grape Belt. The system is designed as a stationary/portable scanning station rather than a moving autonomous rover.
**Limitations:** The current prototype requires the farmer to pluck a leaf and place it in the chamber; it does not perform *in-situ* scanning on the vine. The model is highly specialized to the targeted diseases and may misclassify novel or rare infections. Sensor calibration drifts over extended periods remain a potential maintenance factor.

## 1.5 Organization of Report
The report is organized into six main chapters. Chapter 1 introduces the project. Chapter 2 reviews the existing literature on deep learning in agriculture, IoT, and multi-modal fusion. Chapter 3 details the hardware and software architecture of the proposed system. Chapter 4 explains the implementation details, including dataset curation, model training, and application development. Chapter 5 discusses the results, performance metrics, and evaluation of the system. Finally, Chapter 6 concludes the report and outlines avenues for future work.

---

# CHAPTER 2: LITERATURE REVIEW

## 2.1 Overview of Plant Disease Detection Using Deep Learning
Deep learning, particularly Convolutional Neural Networks (CNNs), has revolutionized automated plant disease detection. Early studies demonstrated the feasibility of using deep CNNs to classify diseases from leaf images with high accuracy. The PlantVillage dataset became a standard benchmark, prompting numerous researchers to apply architectures like AlexNet and VGGNet. While these models achieved accuracies upwards of 95% on held-out test sets from the same distribution, subsequent studies highlighted a critical flaw: models trained on homogeneous laboratory backgrounds suffered significant performance degradation when tested on field-captured images. This established the necessity for models trained on authentic, in-field datasets encompassing realistic variations in illumination, angle, and background clutter.

## 2.2 Transfer Learning Approaches
Given the scarcity of massive, annotated agricultural datasets for specific crops in specific regions, transfer learning has become the standard paradigm. By initializing networks with weights pre-trained on massive datasets like ImageNet, researchers can achieve faster convergence and better generalization on smaller plant disease datasets.
*   **ResNet:** Architectures like ResNet-50 introduced skip connections, allowing for deeper networks without the vanishing gradient problem, achieving excellent accuracy in crop disease tasks but often at the cost of high computational overhead.
*   **EfficientNet:** EfficientNet scaled network depth, width, and resolution systematically, providing an excellent trade-off between accuracy and parameters.
*   **MobileNet:** MobileNet architectures, specifically MobileNetV2 and V3, utilized depthwise separable convolutions and inverted residuals with linear bottlenecks. MobileNetV3 incorporates neural architecture search and squeeze-and-excitation modules, making it exceptionally well-suited for edge devices (like Raspberry Pi) where computational resources and latency are constraints. This efficiency makes it the chosen backbone for LeafLens AI Pro.

## 2.3 Explainability in AI (Grad-CAM)
A major barrier to the adoption of AI in agriculture is the "black box" nature of deep learning models. Agronomists and farmers require transparency to trust algorithmic diagnoses. Gradient-weighted Class Activation Mapping (Grad-CAM) has emerged as a powerful technique to provide visual explanations. By utilizing the gradients of any target concept flowing into the final convolutional layer, Grad-CAM produces a coarse localization map highlighting the important regions in the image for predicting the concept. In plant pathology, ensuring that the model focuses on actual lesions rather than background artifacts is critical for validating model robustness.

## 2.4 IoT in Precision Agriculture
The Internet of Things (IoT) has facilitated precision agriculture by enabling real-time monitoring of diverse environmental parameters. Sensor networks capturing soil moisture, ambient temperature, humidity, and solar radiation provide a continuous stream of data that informs irrigation, fertigation, and pest management. Edge computing, often employing Raspberry Pi or similar microcontrollers, allows for localized data processing, reducing reliance on constant cloud connectivity and minimizing latency—a crucial feature for remote rural farms.

## 2.5 Multi-Modal Sensor Fusion
While computer vision diagnoses *current* symptoms, plant disease progression is fundamentally tied to environmental conditions. Modern approaches are moving towards multi-modal fusion, combining imaging data with time-series environmental data. For fungal diseases like Downy Mildew, high humidity and specific temperature ranges dictate spore germination and infection rates. Algorithms that fuse visual AI confidence with environmental risk factors (e.g., modifying the probability of a fungal outbreak based on high humidity) represent the cutting edge of predictive agronomy, providing a holistic health assessment rather than a simple static classification.

## 2.6 Gap Analysis
Despite extensive research, existing solutions exhibit several gaps:
1.  Over-reliance on synthetic/laboratory datasets leading to poor field generalization.
2.  Purely vision-based approaches ignoring critical micro-climatic context.
3.  Lack of standardized image capture environments in field applications, leading to high variance in input data quality.
4.  Absence of localized, actionable recommendations (often generic advice is given instead of region-specific agricultural institute guidelines).
LeafLens AI Pro addresses these gaps by using a region-specific dataset, introducing a hardware-controlled capture chamber, fusing vision with IoT environmental metrics, and integrating localized ICAR-NRCG spray schedules.

---

# CHAPTER 3: SYSTEM DESIGN & ARCHITECTURE

## 3.1 Overall System Architecture
The LeafLens AI Pro system is designed as a cohesive cyber-physical system. The workflow is as follows:
`Leaf -> IoT Scanning Chamber -> Raspberry Pi Edge Node -> FastAPI Backend -> AI Inference & Fusion -> Streamlit Dashboard`

A user places a grape leaf inside the controlled chamber. The Raspberry Pi activates the standardized lighting and captures a high-resolution image while simultaneously polling the environmental sensors. This data packet (image + sensor telemetry) is transmitted to the local or cloud-hosted FastAPI backend. The AI model performs inference on the image, extracting Grad-CAM maps and necrosis ratios. Concurrently, the multi-modal fusion engine combines the AI outputs with the environmental data to calculate a comprehensive health score and risk multiplier. Finally, these processed insights, along with localized recommendations, are served to the user via the Streamlit dashboard.

## 3.2 Hardware Design
The hardware component is crucial for eliminating the variability that plagues standard plant disease classifiers.
*   **Scanning Chamber:** A custom-designed opaque enclosure that blocks ambient light, ensuring consistent background and illumination for every scan.
*   **Illumination:** A CRI 95+ LED ring light mounted at a 45-degree angle. High Color Rendering Index (CRI) is critical to accurately capture subtle color variations associated with early-stage chlorosis and necrosis. The 45-degree angle minimizes harsh specular reflections on waxy grape leaves.
*   **Environmental Sensors:**
    *   **Sensirion SHT31:** A high-precision I2C temperature and relative humidity sensor placed within the chamber to capture the immediate micro-climate of the leaf sample.
    *   **BH1750:** A digital ambient light sensor used to verify that the chamber is completely sealed from external light before scanning.
*   **Compute Node:** A Raspberry Pi 4 Model B orchestrates the hardware. It interfaces with the sensors via I2C and GPIO.
*   **Camera:** Raspberry Pi Camera Module 3 provides high-resolution, autofocus capabilities necessary for capturing fine structural details of fungal mycelia or bacterial lesions.

## 3.3 Software Architecture
The software stack is built for modularity, speed, and scalability.
*   **Backend (FastAPI):** Chosen for its high performance and automatic interactive API documentation (Swagger UI). It handles incoming scan requests, orchestrates the AI model inference, executes the data fusion logic, and interacts with the database.
*   **Database (SQLite):** A lightweight, serverless relational database used to store historical scan data, environmental telemetry, health scores, and system logs, enabling longitudinal trend analysis.
*   **Frontend (Streamlit):** A rapid-prototyping Python framework used to build the interactive user interface. It features real-time data visualization (Plotly), historical data tables, Grad-CAM overlays, and is localized to support both English and Marathi.

## 3.4 AI Model Architecture
The core vision engine utilizes **MobileNetV3-Large** via transfer learning.
*   **Backbone:** Pre-trained on ImageNet. Its architecture utilizes inverted residual blocks and squeeze-and-excitation modules to extract high-level feature representations efficiently.
*   **Custom Classifier Head:** The original fully connected classification layer is replaced with a custom head designed for our 4-class problem. It consists of:
    *   Global Average Pooling layer.
    *   Linear layer (e.g., 1280 to 256 neurons).
    *   Hardswish activation function.
    *   Dropout layer (e.g., p=0.5) for regularization to prevent overfitting.
    *   Final Linear layer (256 to 4 neurons) outputting raw logits.

## 3.5 Multi-Modal Environmental Fusion Logic
The system does not rely solely on visual prediction. It calculates agronomic metrics to modulate the final risk assessment.
*   **Vapor Pressure Deficit (VPD):** A critical metric for plant transpiration and fungal disease risk.
    *   `VP_sat = 0.61078 * exp((17.27 * T) / (T + 237.3))` (where T is Temperature in °C)
    *   `VPD = VP_sat * (1 - RH / 100)` (where RH is Relative Humidity in %)
*   **Environmental Risk Multiplier:** A rule-based logic engine adjusts the risk based on disease biology. For example, if the AI predicts Downy Mildew with 80% confidence, but the SHT31 sensor reports RH > 85% and optimal temperature (18-24°C), the fusion engine increases the severity rating and triggers an urgent alert, as conditions are highly conducive to rapid sporulation.

## 3.6 Recommendation Engine
The system integrates agronomic knowledge by mapping detected diseases and severity levels to specific, actionable spray schedules recommended by the ICAR-National Research Centre for Grapes (NRCG), Pune. This ensures farmers receive scientifically validated chemical and biological control strategies rather than generic advice.

---

# CHAPTER 4: IMPLEMENTATION

## 4.1 Dataset Collection & Preprocessing
A custom dataset was meticulously curated to ensure real-world applicability in the Nashik region.
*   **Collection:** 2,726 high-resolution images of grape leaves were collected from various vineyards across the Niphad belt over different growth stages.
*   **Classes:** The dataset was strictly categorized by agronomists into four classes: Downy Mildew, Powdery Mildew, Bacterial Leaf Spot, and Healthy.
*   **Splitting:** The dataset was split into Training (70%), Validation (15%), and Testing (15%) sets, ensuring stratification across classes.
*   **Augmentation:** To improve model robustness and prevent overfitting, online data augmentation was implemented during training using PyTorch's `torchvision.transforms`. This included random horizontal and vertical flips, random rotations (up to 45 degrees), slight color jittering (brightness, contrast), and resizing to the 224x224 input resolution expected by MobileNetV3.

## 4.2 Model Training Pipeline
The model was implemented and trained using the PyTorch deep learning framework.
*   **Optimizer:** AdamW (Adam with Weight Decay) was selected for its robust performance and excellent generalization properties, mitigating the overfitting often seen with standard Adam.
*   **Learning Rate Scheduler:** A Cosine Annealing Learning Rate scheduler was employed. This gradually decreases the learning rate following a cosine curve, allowing the model to take larger steps initially and fine-tune its weights as it approaches the optimal minimum.
*   **Loss Function:** Cross-Entropy Loss, suitable for multi-class classification tasks.
*   **Training Parameters:** The network was trained for 15 epochs with a batch size of 32. Only the custom classifier head and the final few convolutional blocks of the MobileNet backbone were fine-tuned, while earlier feature-extraction layers were frozen.

## 4.3 Grad-CAM Explainability Implementation
To generate heatmaps indicating the regions of interest, Grad-CAM was implemented targeting the final convolutional layer of the MobileNetV3 features extractor. During the backward pass, gradients of the predicted class score with respect to the feature map activations are calculated. These gradients are global-average-pooled to obtain neuron importance weights. A weighted combination of the forward activation maps is computed, followed by a ReLU activation to highlight only the positive influences on the class prediction. This resulting heatmap is then upsampled to 224x224 and superimposed onto the original image.

## 4.4 Necrosis Ratio Estimation
An algorithmic approach using OpenCV was implemented to estimate the physical severity of the disease. The image is converted from RGB to the HSV (Hue, Saturation, Value) color space, which is more robust to lighting variations. Careful thresholding is applied to segment non-green areas (brown, yellow, or necrotic black tissues) indicative of lesions or dead tissue.
`Necrosis Ratio = (Pixels in Necrotic Mask) / (Total Leaf Pixels)`
This ratio provides a quantitative measure of physical leaf damage.

## 4.5 Health Score Computation Formula
A unified Health Score (0-100) is calculated to give a quick, holistic view of the plant's status, fusing AI confidence, environmental severity, and physical damage.
`Health Score = 100 - (P_fused(D) * 65 * S_k + NecrosisRatio * 0.35)`
Where:
*   `P_fused(D)` is the AI's probability for the detected disease, modulated by the environmental risk multiplier.
*   `S_k` is a disease-specific severity constant (e.g., Downy Mildew might have a higher base severity than minor spots).
*   `NecrosisRatio` is the percentage of damaged leaf area (0 to 100).
*   The weights (65 and 0.35) were determined empirically to balance the risk of the pathogen presence against actual observed physical damage.

## 4.6 Backend API Development
The FastAPI application exposes several RESTful endpoints:
*   `POST /api/v1/scan`: Accepts multipart/form-data containing the leaf image and JSON payload of sensor data. Orchestrates inference, Grad-CAM, necrosis calculation, and fusion. Saves results to the database and returns a comprehensive JSON response.
*   `GET /api/v1/history`: Retrieves paginated historical scan data from the SQLite database.
*   `GET /api/v1/analytics`: Returns aggregated statistics (e.g., disease distribution over the last 30 days) for dashboard charting.

## 4.7 Database Schema
An SQLite database tracks operations with a primary `scans` table containing 16 columns:
`id, timestamp, image_path, temp_c, humidity_ph, vpd, light_lux, ai_primary_class, ai_confidence, necrosis_ratio, health_score, environment_risk_level, gradcam_path, suggested_action, user_notes, location_id`

## 4.8 Dashboard Development
The Streamlit dashboard acts as the primary user interface. It is designed to be accessible to farmers, featuring a bilingual toggle for English and Marathi. The UI comprises:
*   **Live Scan Dashboard:** Displays current sensor readings, allows image upload or triggering a live capture from the Pi, and presents the inference results (Disease, Confidence, Health Score, Grad-CAM overlay).
*   **Actionable Insights Panel:** Displays the specific ICAR-NRCG chemical spray recommendations based on the detected disease and severity.
*   **Analytics View:** Utilizes Plotly to render interactive charts showing historical disease trends and environmental correlations, aiding in long-term farm management.

## 4.9 Raspberry Pi Edge Firmware
A Python script (`capture_node.py`) runs as a service on the Raspberry Pi. It utilizes the `smbus2` library for I2C communication with the SHT31 and BH1750 sensors, and `RPi.GPIO` to control the LED ring light relay. The script exposes a lightweight local HTTP endpoint that the Streamlit frontend can trigger to initiate a synchronized light-sensor-camera capture sequence.

---

# CHAPTER 5: RESULTS & DISCUSSION

## 5.1 Training Results
The model demonstrated rapid convergence and exceptional accuracy, confirming the efficacy of transfer learning on the curated dataset.

| Epoch | Training Loss | Training Acc. | Validation Loss | Validation Acc. |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 0.4512 | 89.88% | 0.2874 | 92.63% |
| 2 | 0.1834 | 97.59% | 0.1982 | 93.86% |
| **3** | **0.1105** | **98.53%** | **0.0915** | **98.53%** |
| 4 | 0.0892 | 98.85% | 0.1120 | 97.30% |
| 5 | 0.0614 | 99.53% | 0.0984 | 98.03% |

*Note: The best validation accuracy of 98.53% was achieved at Epoch 3. Early stopping mechanisms and careful regularization prevented significant overfitting in subsequent epochs, though validation accuracy fluctuated slightly.*

## 5.2 Confusion Matrix Analysis
Analysis of the confusion matrix on the held-out test set revealed high true positive rates across all classes. The most common (though rare) misclassification occurred between early-stage Powdery Mildew and Healthy leaves with dust or pesticide residue, highlighting the challenge of visual similarity in early pathogen stages and emphasizing the importance of the controlled scanning chamber.

## 5.3 Per-Class Precision, Recall, F1-Score
The model achieved balanced performance across classes, ensuring that no single disease was disproportionately missed or over-predicted.
*   **Healthy:** Precision: 0.99, Recall: 0.99, F1: 0.99
*   **Downy Mildew:** Precision: 0.98, Recall: 0.97, F1: 0.97
*   **Powdery Mildew:** Precision: 0.96, Recall: 0.98, F1: 0.97
*   **Bacterial Leaf Spot:** Precision: 0.98, Recall: 0.98, F1: 0.98
The high recall for Downy Mildew (0.97) is particularly critical, as missing a positive case in a high-humidity environment can lead to rapid outbreak.

## 5.4 Grad-CAM Visual Analysis
Qualitative evaluation of Grad-CAM outputs confirmed the model's reliability. For Downy Mildew images, the heatmaps consistently localized on the characteristic yellowish oil spots on the upper leaf surface or the white fungal growth on the underside. For Bacterial spots, the network focused precisely on the small, dark necrotic lesions surrounded by yellow halos. The model successfully ignored background elements (which were minimized by the chamber) and healthy green tissue.

## 5.5 Multi-Modal Fusion Effectiveness
The integration of environmental data proved highly effective. In field tests, when the AI detected early-stage Downy Mildew with only moderate confidence (e.g., 65%), standard systems might dismiss it. However, the LeafLens fusion engine, detecting concurrent high humidity (RH > 90%) and optimal VPD, elevated the risk profile, resulting in a significantly lower Health Score and an urgent spray recommendation. This dynamic risk assessment models real-world agronomy much more closely than static visual AI.

## 5.6 System Response Time & Performance
The end-to-end latency of the system (from triggering the scan on the Pi to displaying the dashboard result) averages under 2.5 seconds on a local network. The MobileNetV3 inference requires less than 150ms on standard CPU hardware. The system is lightweight enough to be deployed entirely on the edge, ensuring operational capability even in vineyards with poor internet connectivity.

## 5.7 Comparison with Existing Approaches
Compared to generic plant disease apps (e.g., Plantix), LeafLens AI Pro demonstrated significantly higher robustness against false positives caused by varied lighting, thanks to its hardware chamber. Furthermore, unlike models trained purely on PlantVillage data, our model accurately recognized the specific morphological presentations of diseases as they appear in the Nashik region. The inclusion of environmental fusion and Grad-CAM explainability sets this project apart from standard academic implementations.

---

# CHAPTER 6: CONCLUSION & FUTURE WORK

## 6.1 Summary of Contributions
LeafLens AI Pro successfully bridges the gap between deep learning research and practical agricultural application. The project delivered a high-accuracy (98.53%) specialized AI model for grapevine diseases in Nashik. We successfully engineered a hardware scanning chamber integrated with IoT environmental sensors to standardize inputs and capture crucial micro-climatic data. The implementation of a multi-modal fusion engine, combining AI confidence with VPD and environmental risk factors, enables dynamic and reliable health scoring. Finally, the system is encapsulated in a comprehensive, bilingual software ecosystem (FastAPI + Streamlit) that provides agronomically sound, explainable (Grad-CAM), and actionable insights.

## 6.2 Limitations
Current limitations include the necessity of manual leaf plucking for the scanning chamber, which limits throughput. The model is constrained to the four trained classes and requires periodic re-calibration of the environmental sensors to maintain fusion accuracy.

## 6.3 Future Work
*   **Edge Deployment via ONNX:** Quantizing the PyTorch model and converting it to ONNX format to run inference directly on the Raspberry Pi, completely removing the need for a separate backend server and creating a fully standalone device.
*   **Drone Integration:** Adapting the core model architecture to process aerial imagery captured by drones for large-scale, automated vineyard surveillance, shifting from single-leaf analysis to canopy-level health monitoring.
*   **FPO Field Trials:** Conducting extensive longitudinal field trials with local Farmer Producer Organizations (FPOs) in Niphad to quantify the actual reduction in fungicide usage and improvements in yield resulting from the system's adoption.
*   **Expanded Disease Portfolio:** Incrementally updating the dataset and model to include pests (e.g., Mealybugs, Thrips) and nutrient deficiencies (e.g., Magnesium, Potassium deficiency).

---

# REFERENCES

1. Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). Using deep learning for image-based plant disease detection. *Frontiers in plant science*, 7, 1419.
2. Howard, A., Sandler, M., Chu, G., Chen, L. C., Chen, B., Tan, M., ... & Le, Q. V. (2019). Searching for mobilenetv3. In *Proceedings of the IEEE/CVF international conference on computer vision* (pp. 1314-1324).
3. Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-cam: Visual explanations from deep networks via gradient-based localization. In *Proceedings of the IEEE international conference on computer vision* (pp. 618-626).
4. ICAR-National Research Centre for Grapes. (2025). *Advisory for disease and pest management in grapes*. Pune, India.
5. Ray, R. E. A. (2017). Internet of things for smart agriculture: Technologies, practices and future direction. *Journal of Ambient Intelligence and Smart Environments*, 9(4), 395-420.
6. Barbedo, J. G. A. (2018). Factors influencing the use of deep learning for plant disease recognition. *Biosystems engineering*, 172, 84-91.
7. Sladojevic, S., Arsenovic, M., Anderla, A., Culibrk, D., & Stefanovic, D. (2016). Deep neural networks based recognition of plant diseases by leaf image classification. *Computational intelligence and neuroscience*, 2016.
8. Tan, M., & Le, Q. (2019). Efficientnet: Rethinking model scaling for convolutional neural networks. In *International conference on machine learning* (pp. 6105-6114). PMLR.
9. Atoum, Y., Srivastava, S., & Liu, X. (2014). Automatic apple pathology identification. *IEEE Transactions on Automation Science and Engineering*, 12(2), 702-710.
10. Kamilaris, A., & Prenafeta-Boldú, F. X. (2018). Deep learning in agriculture: A survey. *Computers and electronics in agriculture*, 147, 70-90.
11. Tzounis, A., Katsoulas, N., Bartzanas, T., & Kittas, C. (2017). Internet of Things in agriculture, recent advances and future challenges. *Biosystems engineering*, 164, 31-48.
12. Singh, A. K., Ganapathysubramanian, B., Sarkar, S., & Singh, A. (2018). Deep learning for plant stress phenotyping: trends and future perspectives. *Trends in plant science*, 23(10), 883-898.
13. Ferentinos, K. P. (2018). Deep learning models for plant disease detection and diagnosis. *Computers and electronics in agriculture*, 145, 311-318.
14. Fuentes, A., Yoon, S., Kim, S. C., & Park, D. S. (2017). A robust deep-learning-based detector for real-time tomato plant diseases and pests recognition. *Sensors*, 17(9), 2022.
15. Boulent, J., Foucher, S., Théau, J., & St-Charles, P. L. (2019). Convolutional neural networks for the automatic identification of plant diseases. *Frontiers in plant science*, 10, 941.

---

# APPENDIX A: Hardware Bill of Materials

| Item | Specification | Quantity | Purpose |
| :--- | :--- | :--- | :--- |
| Compute Node | Raspberry Pi 4 Model B (4GB) | 1 | Orchestrates sensors, camera, and local API |
| Camera | Raspberry Pi Camera Module 3 | 1 | High-resolution, autofocus leaf imaging |
| Temp/Humidity Sensor | Sensirion SHT31 (I2C) | 1 | Captures micro-climate VPD metrics |
| Light Sensor | BH1750 (I2C) | 1 | Verifies ambient light isolation in chamber |
| Illumination | CRI 95+ LED Ring (5V/12V) | 1 | Standardized 45-degree lighting |
| Relay Module | 5V 1-Channel Relay | 1 | Controls LED ring activation |
| Enclosure | Custom 3D Printed / Opaque Acrylic | 1 | Blocks external light, houses components |
| Power Supply | 5V 3A USB-C (Pi) | 1 | System power |

---

# APPENDIX B: API Endpoint Documentation

**Base URL:** `http://localhost:8000/api/v1`

### 1. `POST /scan`
*   **Description:** Performs a complete AI inference and environmental fusion on a provided leaf image.
*   **Content-Type:** `multipart/form-data`
*   **Parameters:**
    *   `file` (File): The captured leaf image (JPEG/PNG).
    *   `temp_c` (Float): Temperature in Celsius from SHT31.
    *   `humidity_ph` (Float): Relative humidity percentage from SHT31.
    *   `light_lux` (Float): Lux reading from BH1750.
*   **Response (200 OK):** JSON object containing `disease_class`, `ai_confidence`, `health_score`, `vpd`, `necrosis_ratio`, `recommendation_text`, and `gradcam_image_base64`.

### 2. `GET /history`
*   **Description:** Retrieves historical scan records.
*   **Parameters:**
    *   `limit` (Integer, default 50): Number of records to fetch.
    *   `offset` (Integer, default 0): Pagination offset.
*   **Response (200 OK):** JSON array of historical scan objects.

### 3. `GET /analytics/summary`
*   **Description:** Provides aggregated data for dashboard visualizations.
*   **Parameters:**
    *   `days` (Integer, default 30): The lookback period.
*   **Response (200 OK):** JSON object with disease frequency counts, average health scores, and VPD correlation metrics.

---

# APPENDIX C: Full Source Code Listings

*Note: Due to space constraints, only file names and brief descriptions are provided here. Full source code is available in the project repository.*

*   `main.py`: The core FastAPI application, routing, and dependency injection setup.
*   `models/classifier.py`: PyTorch definition of the modified MobileNetV3-Large architecture.
*   `services/inference.py`: Logic for loading weights, preprocessing images, executing model inference, and calculating Grad-CAM.
*   `services/fusion_engine.py`: Agronomic logic calculating VPD, Environmental Risk Multiplier, and the final Health Score.
*   `database/schema.sql`: SQLite table creation definitions.
*   `database/crud.py`: Functions for inserting new scans and retrieving historical data.
*   `frontend/app.py`: Main Streamlit application providing the user interface, routing to different dashboard views.
*   `frontend/views/live_scan.py`: Streamlit logic for the real-time scanning and results display page.
*   `edge/capture_node.py`: Python script running on Raspberry Pi to interface with GPIO, I2C sensors (SHT31, BH1750), and the Pi Camera.
*   `utils/config.py`: Centralized configuration variables (paths, thresholds, database URIs).
*   `train/dataset.py`: PyTorch custom Dataset class for loading and augmenting images.
*   `train/train_loop.py`: Training script with AdamW optimizer, loss calculation, and validation loops.

---
*End of Report*
