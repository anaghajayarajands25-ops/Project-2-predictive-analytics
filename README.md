#  Wildfire Risk Classification using Machine Learning

> Predicting wildfire risk levels — Low, Medium, and High — using satellite imagery and meteorological data from NASA FIRMS.

---

## 🌐 Live Demo

https://wildlife-risk-classification-hmhgxew5ym8wycpsdoegi8.streamlit.app/



---


---

## 📖 Overview

This project applies supervised machine learning to classify wildfire risk levels across geographic regions using real-world satellite and meteorological data. Three models — SVM, Random Forest, and XGBoost — were trained, evaluated, and compared, with **XGBoost identified as the best-performing model** for production use due to its strong generalization and robustness against spatial data splits.

---

## 🎯 Problem Statement

Classify geographic regions into three wildfire risk categories using environmental and meteorological features:

| Risk Level | Description |
|------------|-------------|
| 🟢 Low Risk | Minimal fire activity, low environmental stress |
| 🟡 Medium Risk | Moderate conditions with potential for fire spread |
| 🔴 High Risk | Critical conditions — high temperature, low humidity, elevated FRP |

**Models Applied:**
- Support Vector Machine (SVM)
- Random Forest Classifier
- XGBoost Classifier

**Additional Goal:** Visualize spatial wildfire risk maps and evaluate prediction accuracy across geographic splits.

---

## 📊 Dataset

| Attribute | Details |
|-----------|---------|
| **Name** | NASA FIRMS Dataset |
| **Source** | [NASA Fire Information for Resource Management System](https://firms.modaps.eosdis.nasa.gov/) |
| **Type** | Satellite + Meteorological data |

---

## 🧩 Features Used

| Category | Features |
|----------|----------|
| **Spatial** | Latitude, Longitude |
| **Satellite-derived** | Brightness, Brightness T31, Scan, Track, FRP (Fire Radiative Power) |
| **Detection** | Confidence, Day/Night |
| **Meteorological** | Temperature, Humidity, Wind Speed, Drought Index |

**Target Variable:** `Wildfire Risk` — `Low`, `Medium`, `High`

---

## 🤖 Machine Learning Models

Three models were trained and benchmarked:

1. **Support Vector Machine (SVM)** — Kernel-based classifier, high accuracy but prone to data leakage in this context
2. **Random Forest** — Ensemble of decision trees; interpretable but lower performance
3. **XGBoost** ⭐ — Gradient-boosted trees; best balance of accuracy, generalization, and spatial robustness

---

## 📈 Model Evaluation Results

| Model | Accuracy | F1-Macro | Notes |
|-------|----------|----------|-------|
| SVM | 99.5% | 0.995 | ⚠️ Likely target leakage / overly easy target |
| Random Forest | 77.4% | 0.74 | Lowest performance; simple and interpretable |
| **XGBoost** ⭐ | **92.2%** | **0.922** | **Best model — strong generalization + spatial split** |

### ✅ Why XGBoost is the Best Model

- Achieves **92.2% accuracy** and **F1-Macro of 0.922** — strong across all three risk classes
- Handles **spatial train-test splits** without overfitting (unlike SVM's suspicious 99.5%)
- Robust to feature correlations and missing values common in satellite datasets
- SVM's 99.5% accuracy is flagged as potentially misleading due to target leakage or an overly easy split — XGBoost's more conservative performance is actually a sign of **better real-world reliability**

Evaluation metrics used: Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.

---

## 🔄 Project Workflow

```
Data Collection (NASA FIRMS)
        ↓
Data Preprocessing & Cleaning
        ↓
Feature Engineering
        ↓
Exploratory Data Visualization
        ↓
Train-Test Split (Spatial)
        ↓
Model Training (SVM / Random Forest / XGBoost)
        ↓
Model Evaluation & Comparison
        ↓
Risk Prediction & Map Visualization
```

---

## 🛠️ Technologies Used

| Category | Tools |
|----------|-------|
| **Language** | Python 3 |
| **ML Libraries** | Scikit-learn, XGBoost |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Deployment** | Streamlit *(see Live Demo above)* |
| **Environment** | Google Colab |

---

## 🚀 How to Run

### Option 1: Google Colab

1. Open the notebook in [Google Colab](https://colab.research.google.com/)
2. Upload the NASA FIRMS dataset when prompted
3. Run all cells sequentially (`Runtime → Run All`)
4. View predictions, evaluation metrics, and risk maps inline

### Option 2: Local Setup

```bash
# Clone the repository
git clone https://github.com/your-username/wildfire-risk-classification.git
cd wildfire-risk-classification

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

---

## 📁 Files Included

```
wildfire-risk-classification/
├── notebook/
│   └── wildfire_risk_classification.ipynb   # Main Colab notebook
├── app.py                                    # Streamlit deployment app
├── requirements.txt                          # Python dependencies
└── README.md
```

---

## 🔮 Future Improvements

- [ ] Real-time wildfire monitoring via live satellite feeds
- [ ] Integration with live weather APIs (OpenWeatherMap, NOAA)
- [ ] Improved spatial visualization (Folium / Kepler.gl maps)
- [ ] Deep learning models (LSTM for temporal fire spread prediction)


---

## 👥 Contributors


|Anagha Jayarajan|
|Lekshmi Priya|
|Akshay BS|

---

## 📄 License

This project is for academic and research purposes. Dataset courtesy of [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/).

