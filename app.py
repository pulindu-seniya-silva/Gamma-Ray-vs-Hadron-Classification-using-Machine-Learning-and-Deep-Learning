import streamlit as st
import numpy as np
import tensorflow as tf

# For traditional ML model (e.g. Logistic Regression) saved with joblib:
import joblib
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")


# Feature columns
features = [
    "fLength", "fWidth", "fSize", "fConc", "fConcl", "fAsym",
    "fM3Long", "FM3Trans", "fAlpha", "fDist"
]

# Streamlit UI
st.set_page_config(page_title="Gamma/Hadron Classifier", layout="centered")
st.title("🔮 Gamma or Hadron Particle Classifier")
st.write("Enter the feature values below to predict the class (Gamma = 1, Hadron = 0).")

# Input form
user_input = []
with st.form("prediction_form"):
    for feature in features:
        val = st.number_input(f"{feature}:", min_value=0.0, format="%.4f", key=feature)
        user_input.append(val)
    
    submitted = st.form_submit_button("Predict")

if submitted:
    # Convert to numpy and scale
    X = np.array(user_input).reshape(1, -1)
    X_scaled = scaler.transform(X)

    # Predict
    prediction = model.predict(X_scaled)[0][0]
    class_label = "Gamma 🌟" if prediction >= 0.5 else "Hadron 💥"
    st.success(f"**Prediction:** {class_label} (Confidence: {prediction:.2f})")
