import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing.feature_engineering import feature

data=pd.read_csv("data/aether-iiit-lucknow-stream-flow-prediction-challenge-2026/train_flood.csv")
columns=data.columns
# Load model and scaler
model = load_model("models/model2.keras")
scaler = joblib.load("preprocessing/scaler2.pkl")

st.set_page_config(page_title="Streamflow Prediction", layout="centered")

st.title("🌊 Streamflow Prediction App")
st.write("Enter feature values to predict streamflow")

inputs = {}

for i, features in enumerate(columns[1:-1]):
    inputs[features] = [st.number_input(
        f"{features}",
        key=f"f{i+1}"
    ) ]

# Create dataframe
input_data = pd.DataFrame(
    inputs
)

info=feature(input_data)



if st.button("Predict"):

    st.write(input_data)

    scaled_data = scaler.transform(input_data)

    st.write(scaled_data)

    prediction = model.predict(scaled_data)

    st.write(prediction)

    st.success(f"Predicted Streamflow: {prediction[0][0]:.4f}")