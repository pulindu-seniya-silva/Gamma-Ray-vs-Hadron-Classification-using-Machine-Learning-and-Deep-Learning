import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import pandas as pd

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

model = load_model()
scaler = load_scaler()

# Feature list
features = [
    "fLength", "fWidth", "fSize", "fConc", "fConcl", "fAsym",
    "fM3Long", "FM3Trans", "fAlpha", "fDist"
]

# Get mean and std from the scaler (used for warnings)
means = scaler.mean_
stds = np.sqrt(scaler.var_)

st.set_page_config(page_title="Gamma/Hadron Classifier", layout="centered")
st.title("🔮 Gamma or Hadron Particle Classifier")
st.write("Enter the feature values below to predict the class (Gamma = 1, Hadron = 0).")

user_input = []
out_of_range = False

with st.form("prediction_form"):
    for i, feature in enumerate(features):
        min_range = round(means[i] - 3 * stds[i], 2)
        max_range = round(means[i] + 3 * stds[i], 2)
        val = st.number_input(
            f"{feature} (recommended: {min_range} to {max_range}):",
            format="%.4f", key=feature
        )
        if val < min_range or val > max_range:
            st.warning(f"⚠️ {feature} value {val} is outside the expected range ({min_range} to {max_range}).")
            out_of_range = True
        user_input.append(val)
    
    submitted = st.form_submit_button("Predict")

if submitted:
    X = pd.DataFrame([user_input], columns=features)
    
    try:
        X_scaled = scaler.transform(X)
      #  st.write("🔍 **Scaled input:**", X_scaled)
        st.write("🧾 **Raw input:**", X)
        
        prediction = model.predict(X_scaled)[0][0]
        class_label = "Gamma 🌟" if prediction >= 0.5 else "Hadron 💥"

        if out_of_range:
            st.info("⚠️ One or more inputs are outside normal training range. Prediction may be less accurate.")
        
        st.success(f"**Prediction:** {class_label} (Confidence: {prediction:.2f})")
    
    except Exception as e:
        st.error(f"❌ An error occurred during prediction: {e}")
