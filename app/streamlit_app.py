import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt

# -----------------------
# Page Config
# -----------------------
st.set_page_config(page_title="Obesity Level Predictor", layout="centered")

st.title("🏥 Obesity Level Prediction")
st.markdown("Deep Learning Model with Attention + Focal Loss")

# -----------------------
# Load Model & Objects
# -----------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "saved_model/obesity_model.keras",
        compile=False
    )

@st.cache_resource
def load_objects():
    scaler = joblib.load("saved_model/scaler.pkl")
    label_encoder = joblib.load("saved_model/label_encoder.pkl")
    cat_encoders = joblib.load("saved_model/categorical_encoders.pkl")
    return scaler, label_encoder, cat_encoders

model = load_model()
scaler, label_encoder, cat_encoders = load_objects()

# -----------------------
# Sidebar Inputs
# -----------------------
st.sidebar.header("Enter Patient Details")

age = st.sidebar.slider("Age", 10, 80, 25)
height = st.sidebar.slider("Height (meters)", 1.3, 2.2, 1.7)
weight = st.sidebar.slider("Weight (kg)", 30, 200, 70)

family_history = st.sidebar.selectbox("Family History of Obesity", ["yes", "no"])
smoke = st.sidebar.selectbox("Smoking", ["yes", "no"])
favc = st.sidebar.selectbox("Frequent High Calorie Food", ["yes", "no"])

exercise = st.sidebar.slider("Physical Activity Frequency (0–3)", 0, 3, 1)
screen_time = st.sidebar.slider("Screen Time (0–3)", 0, 3, 1)
water = st.sidebar.slider("Daily Water Intake (1–3)", 1, 3, 2)

# -----------------------
# Create Input DataFrame
# -----------------------
input_df = pd.DataFrame([{
    "Age": age,
    "Height": height,
    "Weight": weight,
    "family_history": family_history,
    "SMOKE": smoke,
    "FAVC": favc,
    "FAF": exercise,
    "TUE": screen_time,
    "CH2O": water
}])

# -----------------------
# Encode Categorical Columns
# -----------------------
for col in cat_encoders:
    input_df[col] = cat_encoders[col].transform(input_df[col])

# -----------------------
# Prediction
# -----------------------
if st.button("Predict Obesity Level"):

    try:
        # Scale input
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = model.predict(input_scaled)

        predicted_class = np.argmax(prediction)
        predicted_label = label_encoder.inverse_transform([predicted_class])[0]
        confidence = np.max(prediction) * 100

        st.success(f"Predicted Level: {predicted_label}")
        st.info(f"Confidence: {confidence:.2f}%")

        # -----------------------
        # Probability Chart
        # -----------------------
        st.subheader("Prediction Probability Distribution")

        probs = prediction[0]
        labels = label_encoder.classes_

        fig = plt.figure()
        plt.bar(labels, probs)
        plt.xticks(rotation=45)
        plt.ylabel("Probability")
        plt.tight_layout()

        st.pyplot(fig)

    except Exception as e:
        st.error("Error during prediction.")
        st.write(str(e))
