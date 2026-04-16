import streamlit as st
import numpy as np
import joblib
import os

# Page config (MUST be first Streamlit command)
st.set_page_config(page_title="Fraud Detection App")

st.title(" Credit Card Fraud Detection System")
st.write("Enter transaction details below:")

# Debug (optional - you can remove later)
st.write("Files in directory:", os.listdir())

# Load model and scalers
try:
    model = joblib.load("fraud_model.pkl")
    scaler_amount = joblib.load("scaler_amount.pkl")
    scaler_time = joblib.load("scaler_time.pkl")
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()   # stop app if loading fails


# Feature Inputs (V1–V28)
st.subheader("Feature Inputs")

input_data = []

col1, col2, col3 = st.columns(3)

for i in range(1, 29):
    if i % 3 == 1:
        val = col1.number_input(f"V{i}", value=0.0)
    elif i % 3 == 2:
        val = col2.number_input(f"V{i}", value=0.0)
    else:
        val = col3.number_input(f"V{i}", value=0.0)

    input_data.append(val)


# Transaction Details

st.subheader("Transaction Details")

amount = st.number_input("Transaction Amount", value=0.0)
time = st.number_input("Transaction Time", value=0.0)

# Prediction
if st.button("Predict Transaction"):

    try:
        # Scale inputs
        scaled_amount = scaler_amount.transform([[amount]])[0][0]
        scaled_time = scaler_time.transform([[time]])[0][0]

        # Add to feature list
        input_data_extended = input_data + [scaled_amount, scaled_time]

        # Convert to array
        input_array = np.array(input_data_extended).reshape(1, -1)

        # Prediction
        prediction = model.predict(input_array)
        probability = model.predict_proba(input_array)[0][1]

        st.subheader("Prediction Result")

        if prediction[0] == 1:
            st.error(" Fraudulent Transaction Detected")
        else:
            st.success(" Normal Transaction")

        st.write(f"Fraud Probability: {round(probability, 4)}")

    except Exception as e:
        st.error(f"Prediction error: {e}")
