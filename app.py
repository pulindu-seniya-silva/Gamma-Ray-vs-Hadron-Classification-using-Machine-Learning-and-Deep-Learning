import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

model = load_model()
scaler = load_scaler()

features = [
    "fLength", "fWidth", "fSize", "fConc", "fConcl", "fAsym",
    "fM3Long", "FM3Trans", "fAlpha", "fDist"
]

st.set_page_config(page_title="Gamma/Hadron Classifier", layout="centered")
st.title("🔮 Gamma or Hadron Particle Classifier")
st.write("Enter the feature values below to predict the class (Gamma = 1, Hadron = 0).")

user_input = []
with st.form("prediction_form"):
    for feature in features:
        val = st.number_input(f"{feature}:", format="%.4f", key=feature)
        user_input.append(val)
    submitted = st.form_submit_button("Predict")

if submitted:
    import pandas as pd 
    X = pd.DataFrame([user_input], columns=features)
    X_scaled = scaler.transform(X)
    st.write("Scaled input:", X_scaled)
    st.write("Raw input:", X)
    

    prediction = model.predict(X_scaled)[0][0]
    class_label = "Gamma 🌟" if prediction >= 0.5 else "Hadron 💥"
    st.success(f"**Prediction:** {class_label} (Confidence: {prediction:.2f})")
