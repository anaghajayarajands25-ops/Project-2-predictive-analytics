import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

st.set_page_config(
    page_title="Wildfire Risk Classification",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Inter:wght@300;400;500&display=swap');

:root {
    --bg:        #0a0e1a;
    --surface:   #111827;
    --card:      #1a2235;
    --border:    #1e3a5f;
    --accent:    #f97316;
    --accent3:   #22d3ee;
    --text:      #e2e8f0;
    --muted:     #64748b;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

.hero {
    background: linear-gradient(135deg, #0a0e1a 0%, #0f2040 50%, #1a0a0a 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2.5rem 2rem 2rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(249,115,22,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.6rem;
    font-weight: 700;
    letter-spacing: 1px;
    background: linear-gradient(90deg, #f97316, #ef4444, #f59e0b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.3rem;
}
.hero-sub {
    font-size: 0.95rem;
    color: var(--muted);
    max-width: 680px;
    line-height: 1.6;
}
.badge {
    display: inline-block;
    background: rgba(249,115,22,0.15);
    border: 1px solid rgba(249,115,22,0.4);
    color: #f97316;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 20px;
    margin-bottom: 0.8rem;
}

.metric-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
.metric-card {
    flex: 1; min-width: 150px;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
}
.metric-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.3rem;
}
.metric-value {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: var(--accent);
}

.section-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--text);
    border-left: 3px solid var(--accent);
    padding-left: 0.75rem;
    margin: 1.5rem 0 1rem;
}

.risk-pill {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 600;
}
.risk-low  { background: rgba(34,197,94,0.15);  color: #22c55e; border: 1px solid rgba(34,197,94,0.3); }
.risk-med  { background: rgba(245,158,11,0.15); color: #f59e0b; border: 1px solid rgba(245,158,11,0.3); }
.risk-high { background: rgba(239,68,68,0.15);  color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }

.info-box {
    background: var(--card);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent3);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    font-size: 0.88rem;
    color: var(--muted);
    line-height: 1.6;
    margin-bottom: 1rem;
}

.coming-soon {
    background: var(--card);
    border: 1px dashed var(--border);
    border-radius: 16px;
    padding: 4rem 2rem;
    text-align: center;
}
.coming-soon-icon { font-size: 3rem; margin-bottom: 1rem; }
.coming-soon-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 0.5rem;
}
.coming-soon-text { color: var(--muted); font-size: 0.9rem; }

[data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 4px !important;
}
[data-baseweb="tab"] {
    border-radius: 8px !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    color: var(--muted) !important;
    padding: 8px 20px !important;
}
[aria-selected="true"] {
    background: rgba(249,115,22,0.15) !important;
    color: var(--accent) !important;
}

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #f97316, #ef4444);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Rajdhani', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    padding: 0.6rem 0;
}

hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# STEP 1 & 2 — Load NASA FIRMS Dataset (from your notebook)
# ══════════════════════════════════════════════════════
@st.cache_data(show_spinner="🛰️  Fetching live NASA FIRMS satellite data…")
def load_data():
    url = "https://firms.modaps.eosdis.nasa.gov/data/active_fire/modis-c6.1/csv/MODIS_C6_1_Global_7d.csv"
    df = pd.read_csv(url)
    # STEP 3 — Basic Cleaning
    df = df.dropna()
    return df

# STEP 4 — Create Risk Label (from your notebook)
def classify_risk(brightness):
    if brightness < 310:
        return 0   # Low
    elif brightness < 330:
        return 1   # Medium
    else:
        return 2   # High

# STEP 6 & 7 — Train-Test Split + Train Random Forest (from your notebook)
@st.cache_resource(show_spinner="🌲  Training Random Forest model…")
def train_model(_df):
    features = ['scan', 'track', 'confidence', 'frp']
    X = _df[features]
    y = _df['risk']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    # STEP 8 — Predictions
    y_pred = rf.predict(X_test)
    return rf, X_test, y_test, y_pred

df = load_data()
df['risk'] = df['brightness'].apply(classify_risk)
rf, X_test, y_test, y_pred = train_model(df)
# STEP 10 — Predict Risk for ALL DATA (for mapping)
df['Predicted_Risk'] = rf.predict(df[['scan', 'track', 'confidence', 'frp']])
# STEP 9 — Evaluation
acc = accuracy_score(y_test, y_pred)

plt.rcParams.update({
    'figure.facecolor': '#111827',
    'axes.facecolor':   '#1a2235',
    'axes.edgecolor':   '#1e3a5f',
    'axes.labelcolor':  '#94a3b8',
    'xtick.color':      '#64748b',
    'ytick.color':      '#64748b',
    'text.color':       '#e2e8f0',
    'grid.color':       '#1e3a5f',
    'grid.linestyle':   '--',
    'grid.alpha':       0.5,
})


# ══════════════════════════════════════════════════════
# HERO BANNER
# ══════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <div class="badge">🛰️ Project #25 &nbsp;|&nbsp; Satellite &amp; Weather Data</div>
    <div class="hero-title">Wildfire Risk Classification</div>
    <p class="hero-sub">
        Classifying regions into <b>Low · Medium · High</b> wildfire risk using
        meteorological features and satellite-derived vegetation indices.
        Powered by NASA FIRMS MODIS C6.1 real-time data.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="metric-row">
    <div class="metric-card">
        <div class="metric-label">🔥 Active Fire Points</div>
        <div class="metric-value">{len(df):,}</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">🌡️ Avg Brightness (K)</div>
        <div class="metric-value">{df['brightness'].mean():.1f}</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">⚡ Avg FRP (MW)</div>
        <div class="metric-value">{df['frp'].mean():.1f}</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">✅ RF Accuracy</div>
        <div class="metric-value">{acc*100:.1f}%</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# SIDEBAR — Manual Prediction
# ══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🔍 Manual Risk Prediction")
    st.markdown("<p style='color:#64748b;font-size:0.82rem'>Random Forest Model — adjust satellite parameters below</p>", unsafe_allow_html=True)
    st.divider()
    scan       = st.number_input("Scan",  min_value=0.0, max_value=10.0,     value=1.0,  step=0.1)
    track      = st.number_input("Track", min_value=0.0, max_value=10.0,     value=1.0,  step=0.1)
    confidence = st.slider("Confidence (%)", 0, 100, 80)
    frp_input  = st.number_input("FRP — Fire Radiative Power (MW)", min_value=0.0, max_value=5000.0, value=50.0)
    st.divider()
    if st.button("🔥  Predict Wildfire Risk"):
        inp  = pd.DataFrame([[scan, track, confidence, frp_input]],
                             columns=['scan', 'track', 'confidence', 'frp'])
        pred = rf.predict(inp)[0]
        pill = {
            0: ("🟢 LOW RISK",    "risk-low"),
            1: ("🟡 MEDIUM RISK", "risk-med"),
            2: ("🔴 HIGH RISK",   "risk-high")
        }
        label, cls = pill[pred]
        st.markdown(f"<div style='text-align:center;margin-top:1rem'>"
                    f"<span class='risk-pill {cls}' style='font-size:1rem;padding:8px 24px'>{label}</span>"
                    f"</div>", unsafe_allow_html=True)
    st.divider()
    st.markdown("<p style='color:#475569;font-size:0.75rem;text-align:center'>"
                "Data: NASA FIRMS MODIS C6.1<br>Updated every few hours</p>",
                unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# MAIN TABS
# ══════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([
    " Random Forest",
    "SVM ",
    "XGBoost"
])

# ── TAB 1 — Random Forest ──────────────────────────────
with tab1:
    st.markdown("<div class='section-title'>Random Forest Classifier — NASA FIRMS</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="info-box">
        <b>Model:</b> Random Forest (100 estimators) &nbsp;|&nbsp;
        <b>Features:</b> scan · track · confidence · frp &nbsp;|&nbsp;
        <b>Risk Labels:</b>
        <span class='risk-pill risk-low'>Low &lt;310K</span>&nbsp;
        <span class='risk-pill risk-med'>Medium 310–330K</span>&nbsp;
        <span class='risk-pill risk-high'>High &gt;330K</span>
        &nbsp;&nbsp;<b>Accuracy:</b>
        <span style='color:#f97316;font-weight:700'>{acc*100:.2f}%</span>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("✅ Accuracy",   f"{acc*100:.2f}%")
    col_b.metric("🌲 Estimators", "100")
    col_c.metric("📐 Test Split", "20%")
    st.divider()

    inner1, inner2, inner3 = st.tabs(["🗺️ Risk Map", "📊 Confusion Matrix", "📋 Classification Report"])

    # STEP 11 — Risk Map (from your notebook)
    with inner1:
        st.markdown("<div class='section-title'>Global Wildfire Risk Map — Last 7 Days</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(14, 6))
        colors = {0: '#22c55e', 1: '#f59e0b', 2: '#ef4444'}
        labels = {0: 'Low',    1: 'Medium',  2: 'High'}
        for level, color in colors.items():
            sub = df[df['Predicted_Risk'] == level]
            ax.scatter(sub['longitude'], sub['latitude'],
                       c=color, label=labels[level], alpha=0.5, s=4, linewidths=0)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("🌍  Wildfire Risk Map  (NASA FIRMS MODIS C6.1)", fontsize=13, pad=14)
        ax.grid(True)
        patches = [mpatches.Patch(color=c, label=l) for l, c in
                   [("Low Risk", '#22c55e'), ("Medium Risk", '#f59e0b'), ("High Risk", '#ef4444')]]
        ax.legend(handles=patches, loc='lower left', framealpha=0.3, fontsize=9)
        fig.tight_layout()
        st.pyplot(fig)

    # Confusion Matrix (from your notebook)
    with inner2:
        st.markdown("<div class='section-title'>Confusion Matrix</div>", unsafe_allow_html=True)
        cm = confusion_matrix(y_test, y_pred)
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Low", "Medium", "High"]
        )
        disp.plot(ax=ax2, colorbar=False, cmap=plt.cm.get_cmap('YlOrRd'))
        ax2.set_title("Confusion Matrix — Wildfire Risk", fontsize=12, pad=12)
        fig2.tight_layout()
        st.pyplot(fig2)

    # Classification Report (from your notebook)
    with inner3:
        st.markdown("<div class='section-title'>Classification Report</div>", unsafe_allow_html=True)
        report = classification_report(y_test, y_pred, target_names=["Low", "Medium", "High"])
        st.code(report, language="text")

    st.markdown("<p style='color:#475569;font-size:0.78rem;margin-top:1rem'>"
                "Data Source: NASA FIRMS MODIS C6.1 Active Fire — refreshed every few hours.</p>",
                unsafe_allow_html=True)


# ── TAB 2 — SVM placeholder ───────────────────────────
with tab2:
    


# ── TAB 3 — XGBoost placeholder ───────────────────────
with tab3:
   
