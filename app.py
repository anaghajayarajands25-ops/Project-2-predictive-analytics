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
import plotly.graph_objects as go

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Wildfire Risk Classification",
    page_icon="🔥",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stButton > button {
        background: linear-gradient(135deg, #ff4500, #ff8c00);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }
    .risk-card {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .risk-low    { background: #1a3a1a; border: 2px solid #2ca25f; color: #2ca25f; }
    .risk-medium { background: #3a2a00; border: 2px solid #fdae6b; color: #fdae6b; }
    .risk-high   { background: #3a0a0a; border: 2px solid #de2d26; color: #de2d26; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# MODEL PATH
# =========================================================
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "models", "wildfire_xgboost_pipeline.joblib")

# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_model():
    loaded_obj = joblib.load(MODEL_PATH)
    model = loaded_obj['pipeline']
    le    = loaded_obj['label_encoder']
    return model, le

model, le = load_model()

# =========================================================
# TITLE
# =========================================================
st.title("🔥 Wildfire Risk Classification System")
st.markdown("""
**Satellite & Weather Based Wildfire Prediction using XGBoost**

Predicts wildfire risk levels — 🟢 Low · 🟠 Medium · 🔴 High —
using NASA FIRMS MODIS satellite observations.
""")
st.divider()

# =========================================================
# SIDEBAR INPUTS
# =========================================================
st.sidebar.header("📌 Input Parameters")

st.sidebar.subheader("🌐 Location")
latitude = st.sidebar.number_input("Latitude",  min_value=-90.0,  max_value=90.0,  value=10.0,  step=0.01)
longitude = st.sidebar.number_input("Longitude", min_value=-180.0, max_value=180.0, value=76.0,  step=0.01)

st.sidebar.subheader("🛰️ Satellite Info")
satellite = st.sidebar.selectbox("Satellite", ["T", "A"], help="T = Terra, A = Aqua")
daynight  = st.sidebar.selectbox("Day / Night", ["D", "N"], help="D = Daytime, N = Nighttime")

st.sidebar.subheader("📡 Scan Parameters")
scan  = st.sidebar.slider("Scan",  min_value=0.5, max_value=5.0, value=1.5, step=0.01)
track = st.sidebar.slider("Track", min_value=0.5, max_value=5.0, value=1.5, step=0.01)

st.sidebar.subheader("🌡️ Thermal & Fire Parameters")
bright_t31 = st.sidebar.slider("Bright T31 (K)", min_value=250.0, max_value=400.0, value=300.0, step=0.1)
frp        = st.sidebar.slider("Fire Radiative Power — FRP (MW)", min_value=0.0, max_value=500.0, value=25.0, step=0.1)
confidence = st.sidebar.slider("Confidence (%)", min_value=0, max_value=100, value=75)

# NOTE: 'brightness' is intentionally excluded — it was used to create the
# target variable (risk_level) during training and is NOT a model feature.
# 'version' is fixed to '6.1NRT' as it is constant across the NASA FIRMS dataset.

# =========================================================
# BUILD INPUT DATAFRAME  (must match training feature columns exactly)
# =========================================================
input_data = pd.DataFrame({
    'latitude':   [latitude],
    'longitude':  [longitude],
    'scan':       [scan],
    'track':      [track],
    'satellite':  [satellite],
    'confidence': [confidence],
    'version':    ['6.1NRT'],   # constant in the training dataset
    'bright_t31': [bright_t31],
    'frp':        [frp],
    'daynight':   [daynight]
})

# =========================================================
# SHOW INPUT DATA
# =========================================================
with st.expander("📋 View Input Data"):
    st.dataframe(input_data, use_container_width=True)

# =========================================================
# PREDICTION
# =========================================================
if st.button("🚀 Predict Wildfire Risk"):

    try:
        # -------------------------------------------------
        # Run prediction
        # -------------------------------------------------
        prediction_enc   = model.predict(input_data)[0]
        prediction_label = le.inverse_transform([prediction_enc])[0]
        probabilities    = model.predict_proba(input_data)[0]

        st.subheader("📊 Prediction Result")

        # -------------------------------------------------
        # Risk level card
        # -------------------------------------------------
        level = prediction_label.lower()
        css_class = f"risk-{level}"
        icons = {"low": "🟢", "medium": "🟠", "high": "🔴"}
        icon  = icons.get(level, "🔥")

        st.markdown(
            f'<div class="risk-card {css_class}">'
            f'{icon} Predicted Risk Level: {prediction_label.upper()}'
            f'</div>',
            unsafe_allow_html=True
        )

        st.divider()

        # -------------------------------------------------
        # Probability bar chart
        # -------------------------------------------------
        st.subheader("📈 Prediction Probabilities")

        prob_df = pd.DataFrame({
            "Risk Level": le.classes_,
            "Probability": probabilities
        })

        color_map = {"high": "#de2d26", "low": "#2ca25f", "medium": "#fdae6b"}
        prob_df["color"] = prob_df["Risk Level"].map(color_map)

        fig = go.Figure(go.Bar(
            x=prob_df["Risk Level"],
            y=prob_df["Probability"],
            marker_color=prob_df["color"],
            text=[f"{p:.2%}" for p in prob_df["Probability"]],
            textposition="outside"
        ))
        fig.update_layout(
            title="Risk Probability Distribution",
            yaxis=dict(title="Probability", range=[0, 1.1]),
            xaxis_title="Risk Level",
            plot_bgcolor="#0f1117",
            paper_bgcolor="#0f1117",
            font_color="#e0e0e0",
            showlegend=False,
            height=380
        )
        st.plotly_chart(fig, use_container_width=True)

        # -------------------------------------------------
        # Map visualization
        # -------------------------------------------------
        st.subheader("🌍 Geographic Visualization")

        map_color = {"low": "green", "medium": "orange", "high": "red"}
        map_df = pd.DataFrame({
            'latitude':  [latitude],
            'longitude': [longitude],
            'risk':      [prediction_label],
            'size':      [20]
        })

        map_fig = px.scatter_mapbox(
            map_df,
            lat="latitude",
            lon="longitude",
            color="risk",
            size="size",
            color_discrete_map=map_color,
            zoom=4,
            mapbox_style="open-street-map",
            title="Predicted Wildfire Location & Risk"
        )
        map_fig.update_layout(
            paper_bgcolor="#0f1117",
            font_color="#e0e0e0",
            height=450
        )
        st.plotly_chart(map_fig, use_container_width=True)

        # -------------------------------------------------
        # Summary metrics
        # -------------------------------------------------
        st.subheader("📌 Input Summary")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Latitude",    f"{latitude:.4f}°")
        col2.metric("Longitude",   f"{longitude:.4f}°")
        col3.metric("FRP (MW)",    f"{frp:.1f}")
        col4.metric("Confidence",  f"{confidence}%")

        col5, col6, col7, col8 = st.columns(4)
        col5.metric("Bright T31",  f"{bright_t31:.1f} K")
        col6.metric("Scan",        f"{scan:.2f}")
        col7.metric("Track",       f"{track:.2f}")
        col8.metric("Day/Night",   daynight)

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        st.info("Please verify the model file is present at `models/wildfire_xgboost_pipeline.joblib`.")

# =========================================================
# ABOUT SECTION
# =========================================================
st.divider()

with st.expander("ℹ️ About This Project"):
    st.markdown("""
## Wildfire Risk Classification using XGBoost

Predicts wildfire risk levels using **NASA FIRMS MODIS Active Fire** satellite data.

### Model
- **Algorithm**: XGBoost (multi-class softprob)
- **Split**: Spatial Group Split (0.5° grid cells) to prevent geographic leakage
- **Accuracy**: ~92.2% | Macro F1: ~0.922

### Features Used by the Model
| Feature | Description |
|---|---|
| `latitude` / `longitude` | Geographic coordinates |
| `scan` / `track` | Along-scan and along-track pixel size |
| `bright_t31` | Channel 31 brightness temperature (K) |
| `frp` | Fire Radiative Power (MW) |
| `confidence` | Detection confidence (%) |
| `satellite` | Terra (T) or Aqua (A) |
| `daynight` | Daytime (D) or Nighttime (N) |
| `version` | Dataset version (fixed: 6.1NRT) |

> **Note**: `brightness` (Channel 21/22) is **not** a model input — it was used
> to derive the `risk_level` target variable (low / medium / high quantile bins)
> and would cause data leakage if included.

### Risk Level Definition
Risk levels were created by binning `brightness` into three equal quantile groups:
- 🟢 **Low** — bottom 33rd percentile
- 🟠 **Medium** — 33rd–66th percentile
- 🔴 **High** — top 33rd percentile
""")
