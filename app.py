import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Hypertension Prediction System",
    page_icon="🫀",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CUSTOM CSS – Clean modern medical UI
# ============================================================
st.markdown("""
<style>
    /* Hide Streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2.5rem;
        max-width: 860px;
    }

    /* Buttons */
    div.stButton > button {
        border-radius: 10px !important;
        min-height: 46px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }

    /* Metrics */
    div[data-testid="stMetric"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 14px 16px;
    }

    /* Cards */
    .info-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.25rem 1.35rem;
        height: 100%;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .info-card h4 {
        margin: 0 0 0.5rem 0;
        font-size: 1.05rem;
        color: #0f172a;
    }
    .info-card p {
        margin: 0;
        color: #475569;
        font-size: 0.9rem;
        line-height: 1.5;
    }

    /* Result cards */
    .result-hypertension {
        background: linear-gradient(135deg, #fef2f2, #fee2e2);
        border: 1.5px solid #f87171;
        border-radius: 16px;
        padding: 1.8rem 1.5rem;
        text-align: center;
        margin: 1rem 0;
    }
    .result-normal {
        background: linear-gradient(135deg, #f0fdf4, #dcfce7);
        border: 1.5px solid #4ade80;
        border-radius: 16px;
        padding: 1.8rem 1.5rem;
        text-align: center;
        margin: 1rem 0;
    }
    .result-label {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 0.3rem;
    }
    .result-title {
        font-size: 1.75rem;
        font-weight: 700;
        margin: 0;
        line-height: 1.25;
    }
    .result-sub {
        font-size: 0.92rem;
        color: #64748b;
        margin-top: 0.4rem;
    }

    /* Disclaimer */
    .disclaimer-box {
        background: #fff7ed;
        border: 1px solid #fdba74;
        border-left: 5px solid #ea580c;
        border-radius: 10px;
        padding: 1rem 1.15rem;
        font-size: 0.88rem;
        color: #9a3412;
        line-height: 1.55;
        margin: 1.2rem 0;
    }

    /* Confidence bar container */
    .conf-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.1rem 1.25rem;
        margin: 1rem 0;
    }

    /* Section headers */
    .section-header {
        font-size: 1.15rem;
        font-weight: 650;
        color: #0f172a;
        margin: 1.4rem 0 0.8rem 0;
    }

    /* Footer */
    .app-footer {
        text-align: center;
        color: #94a3b8;
        font-size: 0.8rem;
        margin-top: 2.5rem;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL
# ============================================================
MODEL_PATH = Path(__file__).parent / "hypertension_random_forest.pkl"

try:
    model = joblib.load(MODEL_PATH)
    model_loaded = True
    model_error = None
except Exception as e:
    model = None
    model_loaded = False
    model_error = str(e)

# ============================================================
# SESSION STATE
# ============================================================
if "page" not in st.session_state:
    st.session_state.page = "Home"
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "confidence" not in st.session_state:
    st.session_state.confidence = None
if "proba_non" not in st.session_state:
    st.session_state.proba_non = None
if "proba_hyp" not in st.session_state:
    st.session_state.proba_hyp = None
if "input_data" not in st.session_state:
    st.session_state.input_data = None
if "show_menu" not in st.session_state:
    st.session_state.show_menu = False

def navigate(page_name: str):
    st.session_state.page = page_name
    st.session_state.show_menu = False

# ============================================================
# TOP HEADER + MENU
# ============================================================
header_left, header_right = st.columns([7, 1])

with header_left:
    st.markdown("### 🫀 Hypertension Prediction System")
    st.caption("Machine Learning Research System · NHANES 2017–2018")

with header_right:
    st.write("")
    if st.button("☰", key="menu_btn", use_container_width=True):
        st.session_state.show_menu = not st.session_state.show_menu

# Navigation menu
if st.session_state.show_menu:
    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        if st.button("🏠 Home", use_container_width=True):
            navigate("Home")
            st.rerun()
    with m2:
        if st.button("🔮 Predict", use_container_width=True):
            navigate("Prediction")
            st.rerun()
    with m3:
        if st.button("📊 Model", use_container_width=True):
            navigate("Model Information")
            st.rerun()
    with m4:
        if st.button("ℹ️ About", use_container_width=True):
            navigate("About")
            st.rerun()
    st.markdown("---")

# ============================================================
# HOME PAGE
# ============================================================
if st.session_state.page == "Home":

    st.markdown("""
    <div class="disclaimer-box">
        <strong>⚠️ Medical Disclaimer</strong> — This is a research and educational tool only. 
        It does <strong>not</strong> provide a medical diagnosis and must never replace professional 
        medical advice, diagnosis, or treatment. Always consult a qualified doctor.
    </div>
    """, unsafe_allow_html=True)

    st.info(
        "A machine learning research system that classifies current hypertension status "
        "using seven selected health and lifestyle factors."
    )

    st.markdown('<div class="section-header">System Overview</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="info-card">
            <h4>🧠 Machine Learning</h4>
            <p>Uses a Random Forest classifier trained on NHANES 2017–2018 data.</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="info-card">
            <h4>📋 Seven Predictors</h4>
            <p>Age, Sex, BMI, High Cholesterol, Diabetes, Smoking History, and Physical Activity.</p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("""
        <div class="info-card">
            <h4>🎯 Prediction Goal</h4>
            <p>Classifies the entered information as Hypertension or Non-Hypertension.</p>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class="info-card">
            <h4>📊 Model Performance</h4>
            <p>ROC-AUC of 76.57% on the held-out test set (research metric only).</p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.write("")

    if st.button("🔮 Start Hypertension Prediction", type="primary", use_container_width=True):
        navigate("Prediction")
        st.rerun()

# ============================================================
# PREDICTION PAGE
# ============================================================
elif st.session_state.page == "Prediction":

    st.markdown("### 🔮 Hypertension Prediction")
    st.write("Enter the required information below to classify current hypertension status.")

    if not model_loaded:
        st.error("The prediction model could not be loaded.")
        st.code(model_error)
        st.stop()

    st.markdown("---")

    # Personal Information
    st.markdown('<div class="section-header">Personal Information</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age (years)", min_value=18, max_value=120, value=40, step=1)
    with col2:
        sex = st.selectbox("Sex", ["Male", "Female"])

    # Health Information
    st.markdown('<div class="section-header">Health Information</div>', unsafe_allow_html=True)
    bmi = st.number_input(
        "BMI (Body Mass Index)",
        min_value=10.0, max_value=80.0, value=25.0, step=0.1,
        help="BMI = weight (kg) / [height (m)]²"
    )
    col3, col4 = st.columns(2)
    with col3:
        cholesterol = st.selectbox("High Cholesterol", ["No", "Yes"])
    with col4:
        diabetes = st.selectbox("Diabetes", ["No", "Yes"])

    # Lifestyle Information
    st.markdown('<div class="section-header">Lifestyle Information</div>', unsafe_allow_html=True)
    col5, col6 = st.columns(2)
    with col5:
        smoking = st.selectbox("Smoking History", ["No", "Yes"])
    with col6:
        physical_activity = st.selectbox("Regular Physical Activity", ["No", "Yes"])

    st.write("")

    if st.button("🔍 Predict Hypertension Status", type="primary", use_container_width=True):
        try:
            # IMPORTANT: Encoding matches the original trained model (0/1)
            input_data = pd.DataFrame([{
                "RIDAGEYR": age,
                "RIAGENDR": 1 if sex == "Male" else 0,
                "BMXBMI": bmi,
                "BPQ080": 1 if cholesterol == "Yes" else 0,
                "DIQ010": 1 if diabetes == "Yes" else 0,
                "SMQ020": 1 if smoking == "Yes" else 0,
                "Physical_Activity": 1 if physical_activity == "Yes" else 0
            }])

            prediction = int(model.predict(input_data)[0])
            proba = model.predict_proba(input_data)[0]
            confidence = float(max(proba)) * 100

            st.session_state.prediction = prediction
            st.session_state.confidence = confidence
            st.session_state.proba_non = float(proba[0]) * 100
            st.session_state.proba_hyp = float(proba[1]) * 100
            st.session_state.input_data = input_data

            navigate("Result")
            st.rerun()

        except Exception as e:
            st.error("Prediction could not be completed.")
            st.write("Please check that the model file matches the application.")
            st.code(str(e))

# ============================================================
# RESULT PAGE
# ============================================================
elif st.session_state.page == "Result":

    st.markdown("### 📊 Prediction Result")

    prediction = st.session_state.prediction
    confidence = st.session_state.confidence

    if prediction is None:
        st.info("No prediction has been made yet.")
        if st.button("🔮 Make a Prediction", use_container_width=True):
            navigate("Prediction")
            st.rerun()
    else:
        # Main result card
        if prediction == 1:
            st.markdown("""
            <div class="result-hypertension">
                <div class="result-label" style="color:#b91c1c;">Elevated Risk Detected</div>
                <div class="result-title" style="color:#991b1b;">HYPERTENSION</div>
                <div class="result-sub">The model indicates a higher likelihood of hypertension</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-normal">
                <div class="result-label" style="color:#15803d;">Lower Risk Detected</div>
                <div class="result-title" style="color:#166534;">NON-HYPERTENSION</div>
                <div class="result-sub">The model indicates a lower likelihood of hypertension</div>
            </div>
            """, unsafe_allow_html=True)

        # Confidence section
        if confidence is not None:
            st.markdown(f"""
            <div class="conf-box">
                <div style="display:flex; justify-content:space-between; margin-bottom:0.45rem;">
                    <span style="font-size:0.9rem; color:#475569; font-weight:500;">Model Confidence</span>
                    <span style="font-size:1rem; font-weight:700; color:#0f172a;">{confidence:.1f}%</span>
                </div>
                <div style="background:#e2e8f0; border-radius:6px; height:11px; overflow:hidden;">
                    <div style="background:{"#ef4444" if prediction==1 else "#22c55e"}; 
                                width:{min(confidence,100)}%; height:100%; border-radius:6px;"></div>
                </div>
                <div style="display:flex; justify-content:space-between; margin-top:0.7rem; 
                            font-size:0.82rem; color:#64748b;">
                    <span>Non-Hypertension: {st.session_state.proba_non:.1f}%</span>
                    <span>Hypertension: {st.session_state.proba_hyp:.1f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Entered information table
        st.markdown('<div class="section-header">Entered Information</div>', unsafe_allow_html=True)

        if st.session_state.input_data is not None:
            data = st.session_state.input_data.iloc[0]
            result_table = pd.DataFrame({
                "Variable": [
                    "Age", "Sex", "BMI", "High Cholesterol",
                    "Diabetes", "Smoking History", "Physical Activity"
                ],
                "Value": [
                    int(data["RIDAGEYR"]),
                    "Male" if data["RIAGENDR"] == 1 else "Female",
                    f"{data['BMXBMI']:.1f}",
                    "Yes" if data["BPQ080"] == 1 else "No",
                    "Yes" if data["DIQ010"] == 1 else "No",
                    "Yes" if data["SMQ020"] == 1 else "No",
                    "Yes" if data["Physical_Activity"] == 1 else "No"
                ]
            })
            st.dataframe(result_table, use_container_width=True, hide_index=True)

        # Strong disclaimer
        st.markdown("""
        <div class="disclaimer-box">
            <strong>⚠️ Important</strong> — This result is for research and educational purposes only. 
            It is <strong>not a medical diagnosis</strong>. Please consult a qualified doctor for proper 
            evaluation and treatment.
        </div>
        """, unsafe_allow_html=True)

        st.write("")
        if st.button("🔄 New Prediction", type="primary", use_container_width=True):
            st.session_state.prediction = None
            st.session_state.confidence = None
            st.session_state.proba_non = None
            st.session_state.proba_hyp = None
            st.session_state.input_data = None
            navigate("Prediction")
            st.rerun()

# ============================================================
# MODEL INFORMATION PAGE
# ============================================================
elif st.session_state.page == "Model Information":

    st.markdown("### 📊 Model Information")
    st.write(
        "The system uses a Random Forest classification algorithm "
        "to classify current hypertension status."
    )

    st.markdown("---")
    st.markdown('<div class="section-header">Dataset</div>', unsafe_allow_html=True)
    st.write("**NHANES 2017–2018** (National Health and Nutrition Examination Survey)")
    st.write("Final usable records: **5,224**")

    st.markdown('<div class="section-header">Algorithm</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-card">
        <h4>🌳 Random Forest</h4>
        <p>An ensemble machine learning method that builds multiple decision trees 
        and combines their votes to produce a more robust classification.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Seven Predictors</div>', unsafe_allow_html=True)
    predictors = pd.DataFrame({
        "No.": [1, 2, 3, 4, 5, 6, 7],
        "Predictor": [
            "Age", "Sex", "BMI", "High Cholesterol",
            "Diabetes", "Smoking History", "Physical Activity"
        ]
    })
    st.dataframe(predictors, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-header">Model Performance (Test Set)</div>', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("Accuracy", "71.29%")
    m2.metric("ROC-AUC", "76.57%")
    m3.metric("F1-Score", "57.26%")

    m4, m5 = st.columns(2)
    m4.metric("Precision", "60.91%")
    m5.metric("Recall", "54.03%")

    st.info(
        "These metrics reflect performance on the held-out test data only. "
        "They should not be interpreted as clinical diagnostic accuracy."
    )

# ============================================================
# ABOUT PAGE
# ============================================================
elif st.session_state.page == "About":

    st.markdown("### ℹ️ About the System")
    st.write(
        "The Hypertension Prediction System is a research-based machine learning "
        "application developed to classify current hypertension status using selected "
        "demographic, health, and lifestyle variables."
    )

    st.markdown("---")
    st.markdown('<div class="section-header">Purpose</div>', unsafe_allow_html=True)
    st.write(
        "To demonstrate how machine learning can be applied to health-related data "
        "for educational and research purposes."
    )

    st.markdown('<div class="section-header">Data Source</div>', unsafe_allow_html=True)
    st.write("National Health and Nutrition Examination Survey (NHANES), 2017–2018.")

    st.markdown('<div class="section-header">Algorithm</div>', unsafe_allow_html=True)
    st.write("Random Forest Classifier.")

    st.markdown("""
    <div class="disclaimer-box">
        <strong>⚠️ Important Notice</strong><br><br>
        This system is intended solely for research and educational use. 
        It is <strong>not a diagnostic device</strong> and has not been approved 
        by any regulatory authority for clinical use.<br><br>
        Results should never replace consultation with a licensed physician. 
        If you have concerns about your blood pressure or cardiovascular health, 
        please seek medical attention.
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div class="app-footer">
    © 2026 Hypertension Prediction System · Machine Learning Research Project<br>
    Random Forest · NHANES 2017–2018 · 7 Predictors
</div>
""", unsafe_allow_html=True)
