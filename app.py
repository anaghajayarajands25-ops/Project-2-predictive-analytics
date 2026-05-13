# =========================================================
# app.py
# Wildfire Risk Classification System (XGBoost)
# Streamlit Deployment App
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Wildfire Risk Classification",
    page_icon="🔥",
    layout="wide"
)

# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_model():
    model = joblib.load("xgboost_wildfire_model.pkl")
    label_encoder = joblib.load("xgboost_label_encoder.pkl")
    return model, label_encoder

model, le = load_model()

# =========================================================
# TITLE
# =========================================================
st.title("🔥 Wildfire Risk Classification System")
st.markdown("""
### Satellite & Weather Based Wildfire Prediction using XGBoost

This system predicts wildfire risk levels:

- 🟢 Low Risk
- 🟠 Medium Risk
- 🔴 High Risk

using NASA FIRMS satellite observations and environmental parameters.
""")

st.divider()

# =========================================================
# SIDEBAR INPUTS
# =========================================================
st.sidebar.header("📌 Input Parameters")

brightness = st.sidebar.slider(
    "Brightness",
    250.0, 500.0, 320.0
)

scan = st.sidebar.slider(
    "Scan",
    0.5, 5.0, 1.5
)

track = st.sidebar.slider(
    "Track",
    0.5, 5.0, 1.5
)

confidence = st.sidebar.slider(
    "Confidence",
    0, 100, 75
)

bright_t31 = st.sidebar.slider(
    "Bright T31",
    250.0, 400.0, 300.0
)

frp = st.sidebar.slider(
    "Fire Radiative Power (FRP)",
    0.0, 500.0, 25.0
)

latitude = st.sidebar.number_input(
    "Latitude",
    value=10.0
)

longitude = st.sidebar.number_input(
    "Longitude",
    value=76.0
)

daynight = st.sidebar.selectbox(
    "Day/Night",
    ["D", "N"]
)

satellite = st.sidebar.selectbox(
    "Satellite",
    ["A", "T"]
)

# =========================================================
# PREPROCESS INPUTS
# =========================================================
daynight_encoded = 1 if daynight == "D" else 0
satellite_encoded = 0 if satellite == "A" else 1

# =========================================================
# CREATE INPUT DATAFRAME
# =========================================================
input_data = pd.DataFrame({
    'latitude': [latitude],
    'longitude': [longitude],
    'brightness': [brightness],
    'scan': [scan],
    'track': [track],
    'satellite': [satellite_encoded],
    'confidence': [confidence],
    'bright_t31': [bright_t31],
    'frp': [frp],
    'daynight': [daynight_encoded]
})

# =========================================================
# PREDICTION
# =========================================================
if st.button("🚀 Predict Wildfire Risk"):

    prediction = model.predict(input_data)[0]
    prediction_label = le.inverse_transform([prediction])[0]

    probabilities = model.predict_proba(input_data)[0]

    st.subheader("📊 Prediction Result")

    # =====================================================
    # RISK DISPLAY
    # =====================================================
    if prediction_label.lower() == "low":
        st.success(f"🟢 Predicted Risk Level: {prediction_label.upper()}")

    elif prediction_label.lower() == "medium":
        st.warning(f"🟠 Predicted Risk Level: {prediction_label.upper()}")

    else:
        st.error(f"🔴 Predicted Risk Level: {prediction_label.upper()}")

    st.divider()

    # =====================================================
    # PROBABILITY CHART
    # =====================================================
    prob_df = pd.DataFrame({
        "Risk Level": le.classes_,
        "Probability": probabilities
    })

    fig = px.bar(
        prob_df,
        x="Risk Level",
        y="Probability",
        text="Probability",
        color="Risk Level",
        title="Prediction Probabilities"
    )

    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')

    st.plotly_chart(fig, use_container_width=True)

    # =====================================================
    # MAP VISUALIZATION
    # =====================================================
    st.subheader("🌍 Location Visualization")

    map_df = pd.DataFrame({
        'latitude': [latitude],
        'longitude': [longitude],
        'risk': [prediction_label]
    })

    map_fig = px.scatter_mapbox(
        map_df,
        lat="latitude",
        lon="longitude",
        color="risk",
        zoom=4,
        size_max=15,
        mapbox_style="open-street-map",
        title="Predicted Wildfire Location"
    )

    st.plotly_chart(map_fig, use_container_width=True)

# =========================================================
# ABOUT SECTION
# =========================================================
st.divider()

with st.expander("ℹ️ About This Project"):
    st.markdown("""
### Wildfire Risk Classification using XGBoost

This project uses:
- NASA FIRMS satellite wildfire data
- Machine Learning (XGBoost)
- Spatial wildfire risk prediction

### Features Used
- Brightness
- FRP (Fire Radiative Power)
- Confidence
- Scan & Track
- Satellite Source
- Day/Night Information
- Geographic Coordinates

### Model
The deployed model is **XGBoost**, selected as the best-performing model after comparing:
- SVM
- Random Forest
- XGBoost

### Authors
Developed for wildfire risk analysis and spatial prediction research.
""")
