import streamlit as st
import numpy as np
import joblib

# load model and scalers using your correct path
model = joblib.load("../results/models/fraud_model.pkl")
scaler_amount = joblib.load("../results/models/scaler_amount.pkl")
scaler_time = joblib.load("../results/models/scaler_time.pkl")

st.set_page_config(page_title="Fraud Detection App")

st.title("Credit Card Fraud Detection System")
st.write("Enter transaction details below:")

input_data = []

st.subheader("Feature Inputs")

col1, col2, col3 = st.columns(3)

for i in range(1, 29):
    if i % 3 == 1:
        val = col1.number_input(f"V{i}", value=0.0)
    elif i % 3 == 2:
        val = col2.number_input(f"V{i}", value=0.0)
    else:
        val = col3.number_input(f"V{i}", value=0.0)
    
    input_data.append(val)

st.subheader("Transaction Details")

amount = st.number_input("Transaction Amount", value=0.0)
time = st.number_input("Transaction Time", value=0.0)

# scaling
scaled_amount = scaler_amount.transform([[amount]])[0][0]
scaled_time = scaler_time.transform([[time]])[0][0]

input_data.append(scaled_amount)
input_data.append(scaled_time)

input_array = np.array(input_data).reshape(1, -1)

if st.button("Predict Transaction"):
    prediction = model.predict(input_array)
    probability = model.predict_proba(input_array)[0][1]

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("Fraudulent Transaction Detected")
    else:
        st.success("Normal Transaction")

    st.write(f"Fraud Probability: {round(probability, 4)}")