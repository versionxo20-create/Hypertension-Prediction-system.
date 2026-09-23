============================================================
  Hypertension Risk Prediction System
  (Improved Streamlit UI + Strong Medical Disclaimer)
============================================================

MODEL
-----
- Algorithm      : Random Forest
- Training data  : NHANES 2017–2018
- Predictors (7) : Age, Sex, BMI, Physical Activity,
                   High Cholesterol, Smoking History, Diabetes Status
- Output         : Hypertension / Non-Hypertension + confidence %

HOW TO RUN LOCALLY
------------------
1. Create a virtual environment (recommended):

   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # macOS / Linux:
   source venv/bin/activate

2. Install dependencies:

   pip install -r requirements.txt

3. Launch the app:

   streamlit run app.py

4. Open the URL shown in the terminal (usually http://localhost:8501)

DEPLOYMENT TIPS
---------------
• Streamlit Community Cloud (easiest free option)
  - Push this folder to a public GitHub repository
  - Go to https://share.streamlit.io and deploy from the repo
  - Make sure the main file is set to "app.py"

• Other options: Render, Railway, Hugging Face Spaces, or any
  server that can run Python + Streamlit

IMPORTANT FILES
---------------
app.py                              → Streamlit application
hypertension_random_forest.pkl      → Trained model (do not rename)
requirements.txt                    → Python dependencies
README.txt                          → This file

MEDICAL DISCLAIMER
------------------
This is a research / educational tool ONLY.
It does NOT provide a medical diagnosis and must never replace
professional medical advice, diagnosis, or treatment.
Always consult a qualified healthcare provider.
