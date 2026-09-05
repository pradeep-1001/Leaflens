import os
import sys
import math
import streamlit as st
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import pandas as pd
from backend.database import get_recent_scans, get_analytics_summary, save_scan

# Reconfigure stdout for Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Streamlit Page Configuration
st.set_page_config(
    page_title="LeafLens AI Pro | Grapevine Health & Disease Monitoring",
    page_icon="🍇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0b3d22 0%, #156637 50%, #1f8a4c 100%);
        color: white;
        padding: 26px 32px;
        border-radius: 18px;
        margin-bottom: 22px;
        box-shadow: 0 12px 28px rgba(11, 61, 34, 0.18);
    }
    
    .input-card {
        background: #ffffff;
        border: 2px dashed #cbd5e1;
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 24px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.03);
    }
    
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.06);
    }
    
    .status-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.88rem;
    }
    .status-healthy { background-color: #dcfce7; color: #15803d; border: 1px solid #86efac; }
    .status-warning { background-color: #fef9c3; color: #a16207; border: 1px solid #fde047; }
    .status-danger { background-color: #fee2e2; color: #b91c1c; border: 1px solid #fca5a5; }

    .stButton>button {
        background: linear-gradient(135deg, #15803d 0%, #166534 100%);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 10px 24px;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        box-shadow: 0 6px 18px rgba(22, 101, 52, 0.3);
        transform: scale(1.01);
    }
</style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_SAMPLES_DIR = os.path.join(BASE_DIR, "test_samples")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Lazy-load Inference Engine
@st.cache_resource
def get_inference_engine():
    from inference import LeafLensDiagnostics
    return LeafLensDiagnostics("best_model.pth")

# Translations (English & Marathi)
TEXTS = {
    "en": {
        "title": "🍇 LeafLens AI Pro",
        "subtitle": "AI-Based Grapevine Disease Detection & Smart Health Monitoring",
        "region": "📍 Niphad & Nashik Grape Belt (Maharashtra, India)",
        "tab_camera": "📸 Take Live Photo",
        "tab_upload": "📁 Upload Photo",
        "tab_samples": "🍇 Pre-loaded Samples",
        "camera_prompt": "Point your camera at a grape leaf and click 'Take Photo'",
        "upload_prompt": "Drag & drop or browse a grape leaf image (JPG, PNG)",
        "chamber_title": "📷 IoT Environmental Telemetry",
        "chamber_desc": "Adjust microclimate telemetry or connect live scanning chamber sensors",
        "temp_label": "Chamber Temperature (°C)",
        "humidity_label": "Relative Humidity (%)",
        "lux_label": "Chamber Illumination (Lux)",
        "wetness_label": "Leaf Surface Dew / Moisture Detected",
        "health_score": "Plant Health Score",
        "diagnosis": "Primary Diagnosis",
        "vpd": "Vapor Pressure Deficit",
        "visual_inspection": "🔬 Visual Symptom Inspection & AI Explainability",
        "original_leaf": "Inspected Grape Leaf",
        "gradcam_view": "AI Grad-CAM Attention Heatmap",
        "necrosis_area": "Estimated Lesion Surface",
        "prob_distribution": "📊 Pathogen Probability Distribution",
        "env_fusion": "🌦️ Multi-Modal Environmental Risk Assessment",
        "recommendations": "📋 Actionable Agronomic Advisory & Spray Schedule (ICAR-NRCG Aligned)",
        "no_image_msg": "📸 Snap a live photo, upload an image, or select a sample leaf to begin AI analysis.",
        "downy_advisory": [
            "⚠️ CRITICAL: Weather is conducive to rapid Downy Mildew (Kevda) sporulation.",
            "Apply systemic fungicide: Metalaxyl 8% + Mancozeb 64% WP @ 2.5 g/L water.",
            "De-sucker lower shoots to minimize soil splash inoculation onto bottom leaves.",
            "Ensure proper vine canopy aeration by judicious summer pruning."
        ],
        "powdery_advisory": [
            "⚠️ Powder-like white fungal spore patches detected on leaf surface (Bhuri).",
            "Spray Wettable Sulfur 80% WDG @ 2.0 g/L or Azoxystrobin 23% SC @ 1.0 mL/L.",
            "Avoid excessive nitrogenous fertilizer which promotes succulent vulnerable shoots.",
            "Ensure sunlight penetration across the fruit and canopy zone."
        ],
        "bacterial_advisory": [
            "⚠️ Angular necrotic water-soaked lesions detected (Bacterial Canker / Karpa).",
            "Spray Copper Oxychloride 50% WP @ 2.5 g/L combined with Streptocycline @ 0.1 g/L.",
            "Disinfect pruning secateurs with 1% sodium hypochlorite.",
            "Burn pruned infected canes to eradicate overwintering bacteria."
        ],
        "healthy_advisory": [
            "✅ Grapevine health is optimal. No active pathogenic fungal or bacterial lesions.",
            "Continue standard preventative micronutrient spray schedule (Zinc, Boron, Magnesium).",
            "Maintain soil moisture at 65-75% field capacity.",
            "Routine weekly canopy monitoring recommended."
        ]
    },
    "mr": {
        "title": "🍇 लीफलेन्स एआय प्रो (LeafLens AI Pro)",
        "subtitle": "द्राक्ष बागेतील रोग निदान आणि स्मार्ट आरोग्य देखरेख प्रणाली",
        "region": "📍 निफाड - नाशिक द्राक्ष पट्टा (महाराष्ट्र)",
        "tab_camera": "📸 थेट फोटो काढा (Camera)",
        "tab_upload": "📁 फोटो अपलोड करा",
        "tab_samples": "🍇 नमुन्यांची यादी",
        "camera_prompt": "द्राक्षाचे पान कॅमेऱ्यासमोर व्यवस्थित धरून 'Take Photo' वर क्लिक करा",
        "upload_prompt": "तुमच्या फोन किंवा कॉम्प्युटरवरून द्राक्षाच्या पानाचा फोटो निवडा",
        "chamber_title": "📷 आयओटी हवामान व सेन्सर्स",
        "chamber_desc": "हवामान बदलून रोगाचा धोका तपासा किंवा सेन्सर डेटा जोडा",
        "temp_label": "तापमान (°C)",
        "humidity_label": "हवेतील आर्द्रता (%)",
        "lux_label": "प्रकाश तीव्रता (Lux)",
        "wetness_label": "पानावरील दव / ओलावा (Dew Detected)",
        "health_score": "द्राक्षवेल आरोग्य गुण (Health Score)",
        "diagnosis": "प्राथमिक रोग निदान",
        "vpd": "बाष्पदाब तूट (VPD)",
        "visual_inspection": "🔬 लक्षणांचे विश्लेषण आणि एआय उष्णता नकाशा (Grad-CAM)",
        "original_leaf": "तपासलेले द्राक्षाचे पान",
        "gradcam_view": "एआय ग्रॅड-कॅम उष्णता नकाशा (रोगट भाग)",
        "necrosis_area": "पानावरील डागांचे प्रमाण",
        "prob_distribution": "📊 रोग शक्यता वितरण",
        "env_fusion": "🌦️ हवामान व वातावरण धोका विश्लेषण",
        "recommendations": "📋 तज्ज्ञ कृषी सल्ला व फवारणी नियोजन (ICAR-NRCG शिफारसी)",
        "no_image_msg": "📸 कृपया कॅमेऱ्याने फोटो काढा, फोटो अपलोड करा किंवा नमुना पान निवडून निदान सुरू करा.",
        "downy_advisory": [
            "⚠️ तातडीचा इशारा: वातावरण डाऊनी मिल्ड्यू (केवडा) प्रसारास अत्यंत अनुकूल आहे.",
            "शिफारस फवारणी: मेटॅलॅक्सिल ८% + मॅन्कोझेब ६४% WP @ २.५ ग्रॅम/लिटर पाणी.",
            "वेलीच्या खालील भागातील अनावश्यक फूट (suckers) काढून टाका जेणेकरून हवेचा प्रवाह चांगला राहील.",
            "बागेमध्ये पाण्याचा निचरा योग्य राहील याची दक्षता घ्या."
        ],
        "powdery_advisory": [
            "⚠️ पावडरी मिल्ड्यू (भुरी) रोगाचे पांढरे बुरशीचे डाग आढळले आहेत.",
            "शिफारस फवारणी: पाण्यात विरघळणारे गंधक (Wettable Sulfur 80% WDG) @ २.० ग्रॅम/लिटर किंवा अझॉक्सीस्ट्रॉबिन २३% SC.",
            "नत्रयुक्त खतांचा अतिवापर टाळा, ज्यामुळे कोवळी फूट कमी होऊन रोगाचा प्रादुर्भाव आटोक्यात राहील.",
            "घड आणि पानांवर सूर्यप्रकाश पोहोचेल अशी कॅनोपी व्यवस्था ठेवा."
        ],
        "bacterial_advisory": [
            "⚠️ जिवाणू करपा (Bacterial Canker) ची लक्षणे आढळून आली आहेत.",
            "शिफारस फवारणी: कॉपर ऑक्सिक्लोराईड ५०% WP @ २.५ ग्रॅम + स्ट्रेप्टोमायसीन @ ०.१ ग्रॅम/लिटर.",
            "छाटणीची अवजारे १% सोडियम हायपोक्लोराईटने निर्जंतुक करा.",
            "छाटलेली रोगट पाने व काड्या बागेबाहेर काढून नष्ट करा."
        ],
        "healthy_advisory": [
            "✅ द्राक्षवेल पूर्णपणे निरोगी आहे. कोणत्याही बुरशी किंवा जिवाणू रोगाची लक्षणे नाहीत.",
            "सूक्ष्म अन्नद्रव्यांची (झिंक, बोरॉन, मॅग्नेशियम) नियमित फवारणी वेळापत्रकानुसार सुरू ठेवा.",
            "जमिनीतील ओलावा वाफसा स्थितीत ठेवा.",
            "दर आठवड्याला बागेची नियमित पाहणी करत राहा."
        ]
    }
}

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/grapes.png", width=68)
    st.markdown("### ⚙️ System Settings")
    lang_choice = st.radio("Language / भाषा निवडा", ["English", "मराठी (Marathi)"], index=0)
    lang = "mr" if "मराठी" in lang_choice else "en"
    t = TEXTS[lang]

    st.markdown("---")
    st.markdown(f"#### {t['chamber_title']}")
    st.caption(t['chamber_desc'])
    
    temp_c = st.slider(t["temp_label"], min_value=12.0, max_value=45.0, value=24.5, step=0.5)
    humidity_pct = st.slider(t["humidity_label"], min_value=20.0, max_value=100.0, value=82.0, step=1.0)
    lux_val = st.slider(t["lux_label"], min_value=100, max_value=2000, value=850, step=50)
    leaf_wetness = st.checkbox(t["wetness_label"], value=True)

    st.markdown("---")
    st.markdown("### 💡 Quick Tip")
    st.info("You can snap a photo directly using your webcam, upload a photo taken on your phone, or test with our pre-loaded Niphad vineyard samples!")

# --- MAIN HEADER ---
st.markdown(f"""
<div class="main-header">
    <h1 style="margin:0; font-size: 2.3rem; font-weight:800;">{t['title']}</h1>
    <p style="margin:6px 0 0 0; font-size: 1.15rem; opacity: 0.95;">{t['subtitle']}</p>
    <p style="margin:8px 0 0 0; font-size: 0.88rem; font-weight:600; opacity: 0.85;">{t['region']}</p>
</div>
""", unsafe_allow_html=True)

# --- PROMINENT INPUT TABS ---
tab_cam, tab_up, tab_sample = st.tabs([t["tab_camera"], t["tab_upload"], t["tab_samples"]])

active_image_path = None

with tab_cam:
    st.markdown(f"##### {t['camera_prompt']}")
    camera_image = st.camera_input("Capture grape leaf", label_visibility="collapsed")
    if camera_image is not None:
        save_path = os.path.join(OUTPUTS_DIR, "live_camera_leaf.jpg")
        with open(save_path, "wb") as f:
            f.write(camera_image.getbuffer())
        active_image_path = save_path
        st.success("📸 Photo captured successfully! Analyzing now...")

with tab_up:
    st.markdown(f"##### {t['upload_prompt']}")
    uploaded_file = st.file_uploader("Upload leaf photo", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    if uploaded_file is not None:
        save_path = os.path.join(OUTPUTS_DIR, "user_uploaded_leaf.jpg")
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        active_image_path = save_path
        st.success("📁 Image uploaded successfully! Analyzing now...")

with tab_sample:
    st.markdown("##### Select a grape leaf from our authentic Niphad, Nashik dataset:")
    col_cls, col_file = st.columns(2)
    sample_classes = ["Downy_Mildew", "Healthy", "Powdery_Mildew", "Bacterial_Leaf_Spot"]
    with col_cls:
        chosen_cls = st.selectbox("Disease Category", sample_classes, index=0)
    
    cls_dir = os.path.join(TEST_SAMPLES_DIR, chosen_cls)
    if os.path.exists(cls_dir):
        sample_files = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        with col_file:
            if sample_files:
                selected_file = st.selectbox("Select Sample Leaf", sample_files, index=0)
                if selected_file and active_image_path is None:
                    active_image_path = os.path.join(cls_dir, selected_file)

# Vapor Pressure Deficit (VPD) calculation in kPa
vp_sat = 0.61078 * math.exp((17.27 * temp_c) / (temp_c + 237.3))
vpd_val = round(vp_sat * (1.0 - (humidity_pct / 100.0)), 2)

# --- EXECUTE DIAGNOSIS ---
if active_image_path and os.path.exists(active_image_path):
    engine = get_inference_engine()
    diag = engine.diagnose(active_image_path)
    
    predicted_disease = diag["predicted_disease"]
    confidence = diag["confidence"]
    health_score = diag["health_score"]
    necrosis_pct = diag["necrosis_percentage"]
    gradcam_path = diag["gradcam_image_path"]
    probs = diag["all_probabilities"]

    # Multi-modal risk evaluation
    env_risk_level = "Normal"
    env_risk_color = "status-healthy"
    
    if humidity_pct > 80 and 18 <= temp_c <= 26 and leaf_wetness:
        if "downy" in predicted_disease.lower():
            health_score = max(5, health_score - 8)
            env_risk_level = "CRITICAL (High Humidity + Fungal Sporulation Zone)"
            env_risk_color = "status-danger"
    elif 20 <= temp_c <= 32 and 45 <= humidity_pct <= 70:
        if "powdery" in predicted_disease.lower():
            env_risk_level = "ELEVATED (Optimal Powdery Mildew Range)"
            env_risk_color = "status-warning"
    else:
        env_risk_level = "LOW (Microclimate Unfavorable for Rapid Spread)"

    if vpd_val < 0.5:
        water_status = "Low (High Humidity / Reduced Transpiration)"
    elif 0.5 <= vpd_val <= 1.4:
        water_status = "Moderate (Ideal Vine Transpiration)"
    else:
        water_status = "High (Dry Air / Rapid Moisture Depletion)"

    # Health score badge styling
    if health_score >= 80:
        health_badge = f'<span class="status-badge status-healthy">🟢 HEALTHY ({health_score}/100)</span>'
    elif health_score >= 50:
        health_badge = f'<span class="status-badge status-warning">🟡 MONITOR CLOSELY ({health_score}/100)</span>'
    else:
        health_badge = f'<span class="status-badge status-danger">🔴 HIGH DISEASE ALERT ({health_score}/100)</span>'

    # Top KPI Metrics Cards
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <p style="color:#64748b; font-size:0.85rem; font-weight:600; margin:0;">{t['health_score']}</p>
            <h2 style="margin:8px 0; font-size:2rem; font-weight:800;">{health_score} <span style="font-size:1rem; font-weight:500; color:#94a3b8;">/ 100</span></h2>
            {health_badge}
        </div>
        """, unsafe_allow_html=True)
    with col2:
        formatted_name = predicted_disease.replace('_', ' ')
        st.markdown(f"""
        <div class="metric-card">
            <p style="color:#64748b; font-size:0.85rem; font-weight:600; margin:0;">{t['diagnosis']}</p>
            <h2 style="margin:8px 0; font-size:1.45rem; font-weight:800; color:#0f172a;">{formatted_name}</h2>
            <span style="font-size:0.9rem; font-weight:700; color:#166534;">{confidence * 100:.1f}% Confidence</span>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <p style="color:#64748b; font-size:0.85rem; font-weight:600; margin:0;">{t['vpd']}</p>
            <h2 style="margin:8px 0; font-size:2rem; font-weight:800; color:#0f172a;">{vpd_val} <span style="font-size:0.95rem; font-weight:600; color:#64748b;">kPa</span></h2>
            <span style="font-size:0.85rem; font-weight:600; color:#475569;">{water_status}</span>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <p style="color:#64748b; font-size:0.85rem; font-weight:600; margin:0;">{t['necrosis_area']}</p>
            <h2 style="margin:8px 0; font-size:2rem; font-weight:800; color:#0f172a;">{necrosis_pct}%</h2>
            <span style="font-size:0.85rem; font-weight:600; color:#64748b;">Chamber: {temp_c}°C | {humidity_pct}% RH</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Visual Inspection & Explainability
    st.markdown(f"### {t['visual_inspection']}")
    vis_col1, vis_col2 = st.columns(2)
    with vis_col1:
        st.markdown(f"**{t['original_leaf']}**")
        st.image(Image.open(active_image_path), use_container_width=True)
    with vis_col2:
        st.markdown(f"**{t['gradcam_view']}**")
        if os.path.exists(gradcam_path):
            st.image(Image.open(gradcam_path), use_container_width=True)

    st.markdown("---")

    # Probability Chart & Multi-Modal Analysis
    chart_col, analysis_col = st.columns([1.1, 0.9])
    
    with chart_col:
        st.markdown(f"### {t['prob_distribution']}")
        classes = list(probs.keys())
        percentages = [probs[c] * 100 for c in classes]
        clean_names = [c.replace('_', ' ') for c in classes]

        fig = go.Figure(go.Bar(
            x=percentages,
            y=clean_names,
            orientation='h',
            marker=dict(
                color=['#ef4444' if 'downy' in c.lower() or 'powdery' in c.lower() or 'bacterial' in c.lower() else '#22c55e' for c in classes],
                line=dict(color='#0f172a', width=1)
            ),
            text=[f"{p:.1f}%" for p in percentages],
            textposition='auto',
        ))
        fig.update_layout(
            height=280,
            margin=dict(l=10, r=20, t=10, b=10),
            xaxis_title="Confidence Percentage (%)",
            xaxis=dict(range=[0, 100]),
            yaxis=dict(autorange="reversed"),
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)

    with analysis_col:
        st.markdown(f"### {t['env_fusion']}")
        st.markdown(f"""
        <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:12px; padding:18px;">
            <p style="margin:0 0 8px 0; font-size:0.95rem;"><b>Microclimate Risk Index:</b> <span class="status-badge {env_risk_color}">{env_risk_level}</span></p>
            <p style="margin:0 0 8px 0; font-size:0.95rem;"><b>Vapor Pressure Deficit (VPD):</b> <code>{vpd_val} kPa</code></p>
            <p style="margin:0 0 8px 0; font-size:0.95rem;"><b>Surface Dew / Wetness:</b> {'💧 Present (High Inoculation Risk)' if leaf_wetness else '☀️ Dry'}</p>
            <p style="margin:0; font-size:0.95rem;"><b>Illumination:</b> <code>{lux_val} Lux</code> (Controlled Chamber Spec)</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Actionable Agronomic Advisory
    st.markdown(f"### {t['recommendations']}")
    if "downy" in predicted_disease.lower():
        advisory_list = t["downy_advisory"]
        border_color = "#ef4444"
    elif "powdery" in predicted_disease.lower():
        advisory_list = t["powdery_advisory"]
        border_color = "#f59e0b"
    elif "bacterial" in predicted_disease.lower():
        advisory_list = t["bacterial_advisory"]
        border_color = "#ef4444"
    else:
        advisory_list = t["healthy_advisory"]
        border_color = "#22c55e"

    bullet_items = "".join([f"<li style='margin-bottom:8px; font-size:0.98rem;'>{item}</li>" for item in advisory_list])
    st.markdown(f"""
    <div style="background:#ffffff; border-left: 6px solid {border_color}; border-radius:10px; padding:20px; box-shadow: 0 4px 12px rgba(0,0,0,0.04);">
        <ul style="margin:0; padding-left:20px; color:#1e293b;">
            {bullet_items}
        </ul>
    </div>
    """, unsafe_allow_html=True)

else:
    st.info(f"👈 {t['no_image_msg']}")

# --- VINEYARD SCAN HISTORY & TEMPORAL TRENDS ---
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("### 📈 Vineyard Scan History & Health Trends")

history = get_recent_scans(limit=25)
if history:
    df_history = pd.DataFrame(history)
    df_history['short_time'] = pd.to_datetime(df_history['timestamp']).dt.strftime('%d-%b %H:%M')
    
    col_hist_chart, col_hist_table = st.columns([1.2, 0.8])
    with col_hist_chart:
        st.markdown("##### Plant Health Score & Humidity Trend")
        fig_trend = go.Figure()
        
        # Health Score Line
        fig_trend.add_trace(go.Scatter(
            x=df_history['short_time'],
            y=df_history['health_score'],
            mode='lines+markers',
            name='Health Score (0-100)',
            line=dict(color='#16a34a', width=3),
            marker=dict(size=8, color='#15803d')
        ))
        
        # Humidity Line
        fig_trend.add_trace(go.Scatter(
            x=df_history['short_time'],
            y=df_history['humidity_pct'],
            mode='lines+markers',
            name='Humidity (%)',
            line=dict(color='#0284c7', width=2, dash='dot'),
            marker=dict(size=6, color='#0369a1')
        ))
        
        fig_trend.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=20, b=20),
            xaxis_title="Scan Timestamp",
            yaxis_title="Score / %",
            yaxis=dict(range=[0, 105]),
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    with col_hist_table:
        st.markdown("##### Recent Vineyard Scans Log")
        display_cols = ['short_time', 'disease', 'health_score', 'humidity_pct']
        clean_df = df_history[display_cols].rename(columns={
            'short_time': 'Time',
            'disease': 'Diagnosis',
            'health_score': 'Health',
            'humidity_pct': 'Humidity %'
        })
        st.dataframe(clean_df, use_container_width=True, hide_index=True, height=270)
else:
    st.caption("No historical scans logged yet. Run a scan or send data via the API to populate historical trends.")

# Footer
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#94a3b8; font-size:0.85rem;">
    LeafLens AI Pro • Smart Viticulture Decision Support System • Niphad, Nashik • Powered by PyTorch & MobileNetV3
</div>
""", unsafe_allow_html=True)
