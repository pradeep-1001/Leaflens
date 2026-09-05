import os
import sys
import uuid
import math
import shutil
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Reconfigure stdout for Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure parent directory is in python path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from inference import LeafLensDiagnostics
from backend.database import init_db, save_scan, get_recent_scans, get_analytics_summary

# Initialize database
init_db()

# Initialize FastAPI App
app = FastAPI(
    title="🍇 LeafLens AI Pro - Core Diagnostics API",
    description="Multi-Modal Grapevine Disease Diagnostic & Environmental Health Monitoring Backend",
    version="1.0.0"
)

# CORS Middleware (Allows connections from Web Dashboard, Mobile App, and IoT Gateways)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static outputs folder for Grad-CAM images
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=OUTPUTS_DIR), name="static")

# Load AI Engine
ai_engine = LeafLensDiagnostics("best_model.pth")

def calculate_vpd(temp_c: float, humidity_pct: float) -> float:
    """Calculates Vapor Pressure Deficit (VPD) in kPa"""
    vp_sat = 0.61078 * math.exp((17.27 * temp_c) / (temp_c + 237.3))
    return round(vp_sat * (1.0 - (humidity_pct / 100.0)), 2)

def generate_icar_advisory(disease: str, temp_c: float, humidity_pct: float, leaf_wetness: bool) -> list:
    """Returns ICAR-NRCG aligned recommendations based on pathogen and weather"""
    recs = []
    d_lower = disease.lower()

    if "downy" in d_lower:
        if humidity_pct > 80 and leaf_wetness:
            recs.append("⚠️ HIGH RISK: Environmental conditions are optimal for rapid Downy Mildew sporulation.")
        recs.append("Apply systemic spray: Metalaxyl 8% + Mancozeb 64% WP @ 2.5 g/L water.")
        recs.append("De-sucker lower shoots to minimize soil splash inoculation onto bottom leaves.")
        recs.append("Ensure proper vine canopy aeration by judicious summer pruning.")
    elif "powdery" in d_lower:
        recs.append("⚠️ Powder-like white fungal spore patches detected on leaf surface (Bhuri).")
        recs.append("Spray Wettable Sulfur 80% WDG @ 2.0 g/L or Azoxystrobin 23% SC @ 1.0 mL/L.")
        recs.append("Avoid excessive nitrogenous fertilizer which promotes succulent vulnerable shoots.")
        recs.append("Ensure sunlight penetration across the fruit and canopy zone.")
    elif "bacterial" in d_lower:
        recs.append("⚠️ Angular necrotic water-soaked lesions detected (Bacterial Canker / Karpa).")
        recs.append("Spray Copper Oxychloride 50% WP @ 2.5 g/L combined with Streptocycline @ 0.1 g/L.")
        recs.append("Disinfect pruning secateurs with 1% sodium hypochlorite.")
        recs.append("Burn pruned infected canes to eradicate overwintering bacteria.")
    else:
        recs.append("✅ Grapevine health is optimal. No active pathogenic fungal or bacterial lesions.")
        recs.append("Continue standard preventative micronutrient spray schedule (Zinc, Boron, Magnesium).")
        recs.append("Maintain soil moisture at 65-75% field capacity.")
    return recs

@app.get("/")
def root():
    return {
        "system": "LeafLens AI Pro",
        "status": "online",
        "docs_url": "/docs",
        "version": "1.0.0",
        "region": "Niphad, Nashik Grape Belt (Maharashtra, India)"
    }

@app.post("/api/v1/scan")
async def process_scan(
    image: UploadFile = File(...),
    device_id: str = Form("CHAMBER-NIPHAD-01"),
    temperature_c: float = Form(24.5),
    humidity_pct: float = Form(80.0),
    lux: float = Form(850.0),
    leaf_wetness: bool = Form(False)
):
    """
    Ingests leaf photo + IoT sensor telemetry, executes AI diagnosis, 
    calculates multi-modal risk, saves to database, and returns diagnostic report.
    """
    scan_id = f"SCAN-{uuid.uuid4().hex[:8].upper()}"
    temp_img_name = f"{scan_id}_input.jpg"
    temp_img_path = os.path.join(OUTPUTS_DIR, temp_img_name)

    # Save uploaded image
    try:
        with open(temp_img_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save image: {str(e)}")

    # 1. Run AI Computer Vision Diagnosis
    ai_result = ai_engine.diagnose(temp_img_path)
    
    disease = ai_result["predicted_disease"]
    confidence = ai_result["confidence"]
    health_score = ai_result["health_score"]
    necrosis_pct = ai_result["necrosis_percentage"]

    # Copy latest Grad-CAM with unique name for historical record
    gradcam_unique_name = f"{scan_id}_gradcam.jpg"
    gradcam_unique_path = os.path.join(OUTPUTS_DIR, gradcam_unique_name)
    if os.path.exists(ai_result["gradcam_image_path"]):
        shutil.copyfile(ai_result["gradcam_image_path"], gradcam_unique_path)

    # 2. Multi-Modal Environmental Risk Fusion
    vpd_kpa = calculate_vpd(temperature_c, humidity_pct)
    env_risk_status = "Normal"

    if humidity_pct > 80 and 18 <= temperature_c <= 26 and leaf_wetness:
        if "downy" in disease.lower():
            health_score = max(5, health_score - 8)
            env_risk_status = "CRITICAL (High Humidity + Fungal Sporulation Zone)"
    elif 20 <= temperature_c <= 32 and 45 <= humidity_pct <= 70:
        if "powdery" in disease.lower():
            env_risk_status = "ELEVATED (Optimal Powdery Mildew Range)"

    # 3. Actionable Agronomic Recommendations
    recommendations = generate_icar_advisory(disease, temperature_c, humidity_pct, leaf_wetness)

    # 4. Save to Persistent SQLite Database
    save_scan(
        scan_id=scan_id,
        device_id=device_id,
        disease=disease,
        confidence=confidence,
        health_score=health_score,
        necrosis_pct=necrosis_pct,
        temperature_c=temperature_c,
        humidity_pct=humidity_pct,
        lux=lux,
        leaf_wetness=leaf_wetness,
        vpd_kpa=vpd_kpa,
        image_path=f"/static/{temp_img_name}",
        gradcam_path=f"/static/{gradcam_unique_name}",
        recommendations=recommendations
    )

    return {
        "scan_id": scan_id,
        "device_id": device_id,
        "disease_diagnosis": disease,
        "confidence": confidence,
        "plant_health_score": health_score,
        "necrosis_percentage": necrosis_pct,
        "microclimate": {
            "temperature_c": temperature_c,
            "humidity_pct": humidity_pct,
            "lux": lux,
            "leaf_wetness": leaf_wetness,
            "vpd_kpa": vpd_kpa,
            "risk_status": env_risk_status
        },
        "media": {
            "input_image_url": f"/static/{temp_img_name}",
            "gradcam_heatmap_url": f"/static/{gradcam_unique_name}"
        },
        "all_probabilities": ai_result["all_probabilities"],
        "recommendations": recommendations
    }

@app.get("/api/v1/history")
def get_scan_history(limit: int = 30):
    """Retrieves recent vineyard scans for trend charts and reports"""
    return get_recent_scans(limit=limit)

@app.get("/api/v1/analytics")
def get_analytics():
    """Returns aggregate vineyard health statistics"""
    return get_analytics_summary()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.server:app", host="0.0.0.0", port=8000, reload=True)
