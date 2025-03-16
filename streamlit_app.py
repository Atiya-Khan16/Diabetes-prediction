import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load("random_forest_model.pkl")

# Title
st.title("Diabetes Prediction App")

# User Inputs
HbA1c_level = st.number_input("HbA1c Level", min_value=3.0, max_value=15.0, value=5.5, step=0.1)
age = st.number_input("Age", min_value=1, max_value=120, value=30, step=1)
blood_glucose_level = st.number_input("Blood Glucose Level", min_value=50, max_value=300, value=100, step=1)
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1)

# Prediction Button
if st.button("Predict"):
    # Convert inputs into a NumPy array (ensure it matches model training order)
    user_data = np.array([[HbA1c_level, age, blood_glucose_level, bmi]])

    # Get Prediction (with threshold adjustment if needed)
    probs = model.predict_proba(user_data)[:, 1]  # Probability for class 1
    adjusted_threshold = 0.3  # Change this threshold if needed
    prediction = (probs >= adjusted_threshold).astype(int)[0]

    # Show Result
    if prediction == 1:
        st.error("⚠️ The model predicts that you may have Diabetes.")
    else:
        st.success("✅ The model predicts that you are NOT Diabetic.")


