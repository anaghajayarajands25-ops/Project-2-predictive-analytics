# =========================================================
# app.py
# Wildfire Risk Classification System (XGBoost)
# Streamlit Deployment App
# =========================================================

import os
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
# MODEL PATHS
# =========================================================
BASE_DIR = os.path.dirname(__file__)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "wildfire_xgboost_pipeline.joblib"
)

ENCODER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "wildfire_label_encoder.joblib"
)

# =========================================================
# LOAD MODEL
# =========================================================
# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_model():

    loaded_obj = joblib.load(
        "models/wildfire_xgboost_pipeline.joblib"
    )

    model = loaded_obj['pipeline']
    le = loaded_obj['label_encoder']

    return model, le

# =========================================================
# TITLE
# =========================================================
st.title("🔥 Wildfire Risk Classification System")

st.markdown("""
### Satellite & Weather Based Wildfire Prediction using XGBoost

This application predicts wildfire risk levels:

- 🟢 Low Risk
- 🟠 Medium Risk
- 🔴 High Risk

using NASA FIRMS satellite wildfire observations and environmental parameters.
""")

st.divider()

# =========================================================
# SIDEBAR INPUTS
# =========================================================
st.sidebar.header("📌 Input Parameters")

latitude = st.sidebar.number_input(
    "Latitude",
    min_value=-90.0,
    max_value=90.0,
    value=10.0
)

longitude = st.sidebar.number_input(
    "Longitude",
    min_value=-180.0,
    max_value=180.0,
    value=76.0
)

brightness = st.sidebar.slider(
    "Brightness",
    250.0,
    500.0,
    320.0
)

scan = st.sidebar.slider(
    "Scan",
    0.5,
    5.0,
    1.5
)

track = st.sidebar.slider(
    "Track",
    0.5,
    5.0,
    1.5
)

confidence = st.sidebar.slider(
    "Confidence",
    0,
    100,
    75
)

bright_t31 = st.sidebar.slider(
    "Bright T31",
    250.0,
    400.0,
    300.0
)

frp = st.sidebar.slider(
    "Fire Radiative Power (FRP)",
    0.0,
    500.0,
    25.0
)

daynight = st.sidebar.selectbox(
    "Day / Night",
    ["D", "N"]
)

satellite = st.sidebar.selectbox(
    "Satellite",
    ["A", "T"]
)

# =========================================================
# CREATE INPUT DATAFRAME
# =========================================================
input_data = pd.DataFrame({
    'latitude': [latitude],
    'longitude': [longitude],
    'brightness': [brightness],
    'scan': [scan],
    'track': [track],
    'satellite': [satellite],
    'confidence': [confidence],
    'bright_t31': [bright_t31],
    'frp': [frp],
    'daynight': [daynight]
})

# =========================================================
# SHOW INPUT DATA
# =========================================================
with st.expander("📋 View Input Data"):
    st.dataframe(input_data)

# =========================================================
# PREDICTION BUTTON
# =========================================================
if st.button("🚀 Predict Wildfire Risk"):

    # =====================================================
    # MODEL PREDICTION
    # =====================================================
    prediction = model.predict(input_data)[0]
    prediction_label = le.inverse_transform([prediction])[0]

    probabilities = model.predict_proba(input_data)[0]

    st.subheader("📊 Prediction Result")

    # =====================================================
    # RISK LEVEL DISPLAY
    # =====================================================
    if prediction_label.lower() == "low":
        st.success(f"🟢 Predicted Risk Level: {prediction_label.upper()}")

    elif prediction_label.lower() == "medium":
        st.warning(f"🟠 Predicted Risk Level: {prediction_label.upper()}")

    else:
        st.error(f"🔴 Predicted Risk Level: {prediction_label.upper()}")

    st.divider()

    # =====================================================
    # PROBABILITY VISUALIZATION
    # =====================================================
    st.subheader("📈 Prediction Probabilities")

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
        title="Risk Probability Distribution"
    )

    fig.update_traces(
        texttemplate='%{text:.2f}',
        textposition='outside'
    )

    fig.update_layout(
        yaxis_title="Probability",
        xaxis_title="Risk Level"
    )

    st.plotly_chart(fig, use_container_width=True)

    # =====================================================
    # MAP VISUALIZATION
    # =====================================================
    st.subheader("🌍 Geographic Visualization")

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
        zoom=3,
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
## Wildfire Risk Classification using XGBoost

This project predicts wildfire risk levels using NASA FIRMS satellite wildfire observations and machine learning techniques.

### Dataset
- NASA FIRMS MODIS Active Fire Dataset

### Machine Learning Models Compared
- Support Vector Machine (SVM)
- Random Forest
- XGBoost

### Final Selected Model
✅ XGBoost was selected as the best-performing model due to:

- High predictive accuracy
- Better spatial generalization
- Strong robustness on unseen regions
- Better handling of nonlinear wildfire patterns

### Features Used
- Brightness
- FRP (Fire Radiative Power)
- Confidence
- Bright T31
- Scan & Track
- Day/Night Information
- Satellite Source
- Geographic Coordinates

### Application
This system can assist in:
- Wildfire monitoring
- Risk assessment
- Environmental analysis
- Spatial wildfire prediction
""")
