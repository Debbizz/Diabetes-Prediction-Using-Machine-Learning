import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image

# Load model, scaler, and encoder
with open("diabetes_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

numeric_cols = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level']
categorical_cols = ['gender', 'smoking_history']
encoded_cols = encoder.get_feature_names_out()

def predict_input(single_input):
    input_df = pd.DataFrame([single_input])
    input_df[numeric_cols] = input_df[numeric_cols].astype(float)
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
    input_df[encoded_cols] = encoder.transform(input_df[encoder.feature_names_in_])
    X_input = input_df[numeric_cols + list(encoded_cols)]
    pred = model.predict(X_input)[0]
    prob = model.predict_proba(X_input)[0][list(model.classes_).index(pred)]
    pred_label = "True" if pred == 1 else "False"
    return pred_label, prob

def main():
    st.title("🩺 Diabetes Prediction App")

    st.markdown("""
    ## Predict diabetes risk based on key health metrics.

    Fill the fields in the sidebar and click **Predict** to get the result.
    """)

    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    age = st.number_input("Age", min_value=1.0)
    hypertension = st.selectbox("Hypertension", [0, 1])
    heart_disease = st.selectbox("Heart Disease", [0, 1])
    smoking_history = st.selectbox("Smoking History", ["never", "former", "current", "not current", "unknown"])
    bmi = st.number_input("BMI")
    hba1c_level = st.number_input("HbA1c Level")
    blood_glucose_level = st.number_input("Blood Glucose Level")

    if st.button("Predict"):
        new_input = {
            'gender': gender,
            'age': age,
            'hypertension': hypertension,
            'heart_disease': heart_disease,
            'smoking_history': smoking_history,
            'bmi': bmi,
            'HbA1c_level': hba1c_level,
            'blood_glucose_level': blood_glucose_level
        }
        pred, prob = predict_input(new_input)
        st.success(f"Prediction: **{pred}**")
        st.info(f"Probability: **{prob:.2f}**")

if __name__ == '__main__':
    main()
