import streamlit as st
import pandas as pd
import pickle
import numpy as np

# 1. Load the saved model and encoders
with open('loan_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

st.set_page_config(page_title="Loan Risk Predictor", layout="centered")

st.title("🏦 Loan Default Risk Predictor")
st.markdown("This app predicts whether a loan application will be **Approved** or **Rejected** based on applicant details.")

st.divider()

# 2. Input Section
st.subheader("Applicant Information")
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])

with col2:
    applicant_income = st.number_input("Applicant Monthly Income ($)", min_value=0, value=5000)
    coapplicant_income = st.number_input("Co-applicant Monthly Income ($)", min_value=0, value=0)
    loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0, value=150)
    loan_term = st.selectbox("Loan Term (Days)", [360, 180, 480, 300, 240, 120, 84, 60, 36, 12])
    credit_history = st.selectbox("Credit History", ["Clear (1.0)", "Not Clear (0.0)"])

property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# 3. Pre-processing the inputs
if st.button("Predict Loan Status", type="primary"):
    
    # Convert Credit History to numeric
    ch = 1.0 if "Clear" in credit_history else 0.0
    
    # Create a dictionary for the inputs
    input_data = {
        'Gender': gender,
        'Married': married,
        'Dependents': dependents,
        'Education': education,
        'Self_Employed': self_employed,
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': float(loan_term),
        'Credit_History': ch,
        'Property_Area': property_area,
        'TotalIncome': applicant_income + coapplicant_income # Feature Engineering
    }
    
    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Apply Encoding to categorical columns
    for col in ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']:
        le = encoders[col]
        input_df[col] = le.transform(input_df[col])
        
    # Reorder columns to match the model training order
    column_order = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 
                    'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
                    'Loan_Amount_Term', 'Credit_History', 'Property_Area', 'TotalIncome']
    input_df = input_df[column_order]

    # 4. Prediction
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[0][1]

    st.divider()
    if prediction[0] == 1:
        st.success(f"### Result: APPROVED ✅")
        st.write(f"Confidence Level: {probability:.2%}")
    else:
        st.error(f"### Result: REJECTED ❌")
        st.write(f"Confidence Level: {(1-probability):.2%}")