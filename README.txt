============================================================
  Hypertension Prediction System
  (Improved Multi-Page Version)
============================================================

FEATURES
--------
• Multi-page navigation (Home, Prediction, Result, Model Info, About)
• Clean modern medical UI
• Strong medical disclaimers on key pages
• Correct 0/1 feature encoding matching the trained model
• Confidence score with visual progress bar
• Summary table of entered values on the Result page
• Model performance metrics page

HOW TO RUN
----------
1. Unzip this folder

2. Create a virtual environment (recommended):
   python -m venv venv

   Windows:
   venv\Scripts\activate

   macOS / Linux:
   source venv/bin/activate

3. Install dependencies:
   pip install -r requirements.txt

4. Launch the app:
   streamlit run app.py

5. Open the URL shown in the terminal (usually http://localhost:8501)

FILES INCLUDED
--------------
app.py                              → Main Streamlit application
hypertension_random_forest.pkl      → Trained Random Forest model
requirements.txt                    → Python dependencies
README.txt                          → This file

IMPORTANT
---------
Feature encoding used by the model:
  Sex (RIAGENDR)           : Male = 1, Female = 0
  High Cholesterol (BPQ080): Yes = 1, No = 0
  Diabetes (DIQ010)        : Yes = 1, No = 0
  Smoking (SMQ020)         : Yes = 1, No = 0
  Physical Activity        : Yes = 1, No = 0

MEDICAL DISCLAIMER
------------------
This is a research / educational tool ONLY.
It does NOT provide a medical diagnosis and must never replace
professional medical advice, diagnosis, or treatment.
Always consult a qualified healthcare provider.
