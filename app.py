import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Hypertension Prediction System",
                   page_icon="❤️", layout="centered")

MODEL_PATH = Path(__file__).parent / "hypertension_random_forest.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

st.title("Hypertension Prediction System")
st.write("Enter the required health information to predict hypertension status.")

st.subheader("Patient Information")

age = st.number_input("Age (years)", 18, 120, 40)
sex = st.selectbox("Sex", ["Male", "Female"])
bmi = st.number_input("BMI", 10.0, 80.0, 25.0, step=0.1)
cholesterol = st.selectbox("High Cholesterol", ["No", "Yes"])
diabetes = st.selectbox("Diabetes", ["No", "Yes"])
smoking = st.selectbox("Smoking History", ["No", "Yes"])
activity = st.selectbox("Physical Activity", ["No", "Yes"])

if st.button("Predict Hypertension Status", use_container_width=True):
    X_new = pd.DataFrame([{
        "RIDAGEYR": age,
        "RIAGENDR": 1 if sex == "Male" else 0,
        "BMXBMI": bmi,
        "BPQ080": 1 if cholesterol == "Yes" else 0,
        "DIQ010": 1 if diabetes == "Yes" else 0,
        "SMQ020": 1 if smoking == "Yes" else 0,
        "Physical_Activity": 1 if activity == "Yes" else 0
    }])

    prediction = int(model.predict(X_new)[0])
    probabilities = model.predict_proba(X_new)[0]
    confidence = probabilities[prediction] * 100

    st.divider()
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("HYPERTENSION")
    else:
        st.success("NON-HYPERTENSION")

    st.metric("Model confidence", f"{confidence:.2f}%")
    st.info("This system is a research project and does not replace professional medical diagnosis.")

st.divider()
st.caption("Random Forest | NHANES 2017–2018 | 7 predictors")
