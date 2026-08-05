import streamlit as st
import joblib
import pandas as pd

# ----------------------------
# Load Model, Scaler & Encoders
# ----------------------------
model = joblib.load("models/diabetes_model.pkl")
scaler = joblib.load("models/scaler.pkl")
gender_encoder = joblib.load("models/gender_encoder.pkl")
smoking_encoder = joblib.load("models/smoking_encoder.pkl")

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Diabetes Prediction System")
st.write("Enter the patient's details below.")

# ----------------------------
# User Inputs
# ----------------------------

gender = st.selectbox(
    "Gender",
    list(gender_encoder.classes_)
)

age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=25
)

hypertension = st.selectbox(
    "Hypertension",
    [0, 1]
)

heart_disease = st.selectbox(
    "Heart Disease",
    [0, 1]
)

smoking_history = st.selectbox(
    "Smoking History",
    list(smoking_encoder.classes_)
)

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=70.0,
    value=22.0
)

hba1c = st.number_input(
    "HbA1c Level",
    min_value=3.0,
    max_value=15.0,
    value=5.2
)

blood_glucose = st.number_input(
    "Blood Glucose Level",
    min_value=50,
    max_value=400,
    value=90
)

# ----------------------------
# Prediction
# ----------------------------

if st.button("Predict"):

    input_data = pd.DataFrame([{
        "gender": gender_encoder.transform([gender])[0],
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "smoking_history": smoking_encoder.transform([smoking_history])[0],
        "bmi": bmi,
        "HbA1c_level": hba1c,
        "blood_glucose_level": blood_glucose
    }])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]

    confidence = model.predict_proba(input_scaled)[0].max() * 100

    if prediction == 1:
        st.error("Prediction: Diabetic")
    else:
        st.success("Prediction: Non-Diabetic")

    st.info(f"Confidence Score: {confidence:.2f}%")