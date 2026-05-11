import streamlit as st

st.set_page_config(page_title="Wildfire Risk App", page_icon="🔥", layout="wide")
st.title("🔥 Wildfire Risk Prediction App")

# Create 3 tabs


tab1, tab2, tab3 = st.tabs(["🌿 Random Forest ", "📌 Model 2-xgboost", "📌 Model 3-SVM"])

# ── LOAD DATA ──────────────────────────────────────────────────
@st.cache_data(show_spinner="Fetching live NASA FIRMS data...")
def load_data():
    url = "https://firms.modaps.eosdis.nasa.gov/data/active_fire/modis-c6.1/csv/MODIS_C6_1_Global_7d.csv"
    df = pd.read_csv(url)
    df = df.dropna()
    return df

def classify_risk(brightness):
    if brightness < 310:
        return 0
    elif brightness < 330:
        return 1
    else:
        return 2

@st.cache_resource(show_spinner="Training Random Forest model...")
def train_model(_df):
    features = ['scan', 'track', 'confidence', 'frp']
    X = _df[features]
    y = _df['risk']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    return rf, X_test, y_test, y_pred

df = load_data()
df['risk'] = df['brightness'].apply(classify_risk)
rf, X_test, y_test, y_pred = train_model(df)

# ══════════════════════════════════════════════════════════════
with tab1:
    st.subheader("🔥 Wildfire Risk Predictor — NASA FIRMS (MODIS)")
    st.markdown("Uses **live NASA FIRMS satellite data** + Random Forest to classify wildfire risk.")

    # Sidebar manual prediction
    st.sidebar.header("🔍 Predict Risk Manually")
    scan = st.sidebar.number_input("Scan", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    track = st.sidebar.number_input("Track", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    confidence = st.sidebar.slider("Confidence (%)", 0, 100, 80)
    frp = st.sidebar.number_input("FRP (Fire Radiative Power)", min_value=0.0, max_value=5000.0, value=50.0)

    if st.sidebar.button("🔥 Predict Risk"):
        input_data = pd.DataFrame([[scan, track, confidence, frp]],
                                   columns=['scan', 'track', 'confidence', 'frp'])
        pred = rf.predict(input_data)[0]
        labels = {0: "🟢 Low", 1: "🟡 Medium", 2: "🔴 High"}
        st.sidebar.success(f"Predicted Risk: **{labels[pred]}**")

    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Fire Points", len(df))
    col2.metric("Model Accuracy", f"{accuracy_score(y_test, y_pred)*100:.2f}%")
    col3.metric("Features Used", "scan, track, confidence, frp")

    st.divider()

    # Inner tabs
    inner_tab1, inner_tab2, inner_tab3 = st.tabs(["🗺️ Risk Map", "📊 Confusion Matrix", "📋 Classification Report"])

    with inner_tab1:
        df['Predicted_Risk'] = rf.predict(df[['scan', 'track', 'confidence', 'frp']])
        risk_colors = {0: 'green', 1: 'orange', 2: 'red'}
        risk_labels = {0: 'Low', 1: 'Medium', 2: 'High'}
        fig, ax = plt.subplots(figsize=(12, 6))
        for risk_level, color in risk_colors.items():
            subset = df[df['Predicted_Risk'] == risk_level]
            ax.scatter(subset['longitude'], subset['latitude'],
                       c=color, label=risk_labels[risk_level], alpha=0.5, s=5)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("🌍 Wildfire Risk Map (NASA FIRMS — Last 7 Days)")
        ax.legend(title="Risk Level")
        st.pyplot(fig)

    with inner_tab2:
        cm = confusion_matrix(y_test, y_pred)
        fig2, ax2 = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Low", "Medium", "High"])
        disp.plot(ax=ax2)
        ax2.set_title("Confusion Matrix — Wildfire Risk")
        st.pyplot(fig2)

    with inner_tab3:
        report = classification_report(y_test, y_pred, target_names=["Low", "Medium", "High"])
        st.code(report)

    st.caption("Data Source: NASA FIRMS MODIS C6.1 — Updated every few hours.")

# ══════════════════════════════════════════════════════════════

with tab2:
    # PASTE TEAMMATE 2's app.py CODE HERE
    pass

with tab3:
    # PASTE TEAMMATE 3's app.py CODE HERE
    pass
