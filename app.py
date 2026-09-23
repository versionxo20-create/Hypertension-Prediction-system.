import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="Hypertension Risk Predictor",
    page_icon="❤️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# -------------------------------------------------
# Custom CSS – modern clean medical UI
# -------------------------------------------------
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Main container */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 780px;
    }

    /* Title */
    .main-title {
        font-size: 1.9rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.15rem;
        letter-spacing: -0.3px;
    }
    .subtitle {
        color: #64748b;
        font-size: 0.95rem;
        margin-bottom: 1.4rem;
    }

    /* Disclaimer */
    .disclaimer {
        background: #fff7ed;
        border: 1px solid #fdba74;
        border-left: 5px solid #ea580c;
        border-radius: 10px;
        padding: 1rem 1.15rem;
        font-size: 0.88rem;
        color: #9a3412;
        line-height: 1.55;
        margin-bottom: 1.6rem;
    }

    /* Section title */
    .section-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 1.0rem;
    }

    /* Result boxes */
    .result-positive {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border: 1.5px solid #f87171;
        border-radius: 14px;
        padding: 1.6rem 1.4rem;
        text-align: center;
        margin: 0.8rem 0 1.2rem 0;
    }
    .result-negative {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 1.5px solid #4ade80;
        border-radius: 14px;
        padding: 1.6rem 1.4rem;
        text-align: center;
        margin: 0.8rem 0 1.2rem 0;
    }
    .result-label {
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }
    .result-value {
        font-size: 1.7rem;
        font-weight: 700;
        margin: 0;
        line-height: 1.2;
    }
    .result-sub {
        font-size: 0.9rem;
        color: #64748b;
        margin-top: 0.45rem;
    }

    /* Confidence bar */
    .conf-container {
        background: #f8fafc;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
        border: 1px solid #e2e8f0;
    }

    /* Footer */
    .app-footer {
        text-align: center;
        color: #94a3b8;
        font-size: 0.8rem;
        margin-top: 2rem;
        line-height: 1.6;
    }

    /* Better selectbox / input styling */
    div[data-baseweb="select"] > div {
        border-radius: 8px !important;
    }
    .stNumberInput input {
        border-radius: 8px !important;
    }

    /* Button */
    .stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        height: 2.9rem !important;
        font-size: 1rem !important;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Load model
# -------------------------------------------------
MODEL_PATH = Path(__file__).parent / "hypertension_random_forest.pkl"

@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Could not load the model file: {e}")
        st.stop()

model = load_model()

# -------------------------------------------------
# Header
# -------------------------------------------------
st.markdown('<div class="main-title">❤️ Hypertension Risk Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">AI-powered risk assessment using Random Forest · NHANES 2017–2018</div>',
    unsafe_allow_html=True
)

# -------------------------------------------------
# Strong disclaimer (always visible)
# -------------------------------------------------
st.markdown("""
<div class="disclaimer">
    <strong>⚠️ Medical Disclaimer</strong> — This is a research and educational tool only. 
    It does <strong>not</strong> provide a medical diagnosis and is not a substitute for professional 
    medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider.
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Input section
# -------------------------------------------------
st.markdown('<div class="section-title">📋 Patient Details</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="medium")

with col1:
    age = st.number_input("Age (years)", min_value=18, max_value=120, value=42, step=1)
    sex = st.selectbox("Sex", ["Male", "Female"])
    bmi = st.number_input("BMI", min_value=12.0, max_value=60.0, value=24.5, step=0.1,
                          help="Body Mass Index = weight(kg) / height(m)²")
    activity = st.selectbox("Regular Physical Activity", ["No", "Yes"])

with col2:
    chol = st.selectbox("High Cholesterol", ["No", "Yes"])
    diabetes = st.selectbox("Diabetes", ["No", "Yes"])
    smoking = st.selectbox("Smoking History", ["No", "Yes"])

st.write("")  # small gap

# -------------------------------------------------
# Predict button
# -------------------------------------------------
predict = st.button("Predict Risk", use_container_width=True, type="primary")

# -------------------------------------------------
# Results
# -------------------------------------------------
if predict:
    X = pd.DataFrame([{
        "RIDAGEYR": age,
        "RIAGENDR": 1 if sex == "Male" else 0,
        "BMXBMI": bmi,
        "BPQ080": 1 if chol == "Yes" else 0,
        "DIQ010": 1 if diabetes == "Yes" else 0,
        "SMQ020": 1 if smoking == "Yes" else 0,
        "Physical_Activity": 1 if activity == "Yes" else 0
    }])

    try:
        pred = int(model.predict(X)[0])
        proba = model.predict_proba(X)[0]
        conf = float(proba[pred]) * 100
        conf_non = float(proba[0]) * 100
        conf_hyp = float(proba[1]) * 100

        st.markdown("---")
        st.markdown('<div class="section-title">📊 Prediction Result</div>', unsafe_allow_html=True)

        if pred == 1:
            st.markdown(f"""
            <div class="result-positive">
                <div class="result-label" style="color:#b91c1c;">Elevated Risk</div>
                <div class="result-value" style="color:#991b1b;">Hypertension Risk</div>
                <div class="result-sub">The model suggests higher likelihood of hypertension</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-negative">
                <div class="result-label" style="color:#15803d;">Lower Risk</div>
                <div class="result-value" style="color:#166534;">Non-Hypertension</div>
                <div class="result-sub">The model suggests lower likelihood of hypertension</div>
            </div>
            """, unsafe_allow_html=True)

        # Confidence section
        st.markdown(f"""
        <div class="conf-container">
            <div style="display:flex; justify-content:space-between; margin-bottom:0.4rem;">
                <span style="font-size:0.9rem; color:#475569; font-weight:500;">Model Confidence</span>
                <span style="font-size:0.95rem; font-weight:700; color:#0f172a;">{conf:.1f}%</span>
            </div>
            <div style="background:#e2e8f0; border-radius:6px; height:10px; overflow:hidden;">
                <div style="background:{"#ef4444" if pred==1 else "#22c55e"}; width:{conf}%; height:100%; border-radius:6px;"></div>
            </div>
            <div style="display:flex; justify-content:space-between; margin-top:0.7rem; font-size:0.82rem; color:#64748b;">
                <span>Non-Hypertension: {conf_non:.1f}%</span>
                <span>Hypertension: {conf_hyp:.1f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Soft reminder
        st.info("Remember: This result is for educational purposes only. Please consult a doctor for real medical advice.", icon="ℹ️")

    except Exception as e:
        st.error(f"Prediction error: {e}")

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("""
<div class="app-footer">
    Random Forest Classifier &nbsp;·&nbsp; Trained on NHANES 2017–2018<br>
    Predictors: Age · Sex · BMI · Physical Activity · High Cholesterol · Smoking · Diabetes
</div>
""", unsafe_allow_html=True)
