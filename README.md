# Project-2-predictive-analytics
Wildlife risk classification from satellite and weather data

1.Project Overview

This project focuses on predicting wildfire risk levels (Low, Medium, High) using meteorological and satellite-derived data. By applying machine learning models, we aim to identify high-risk regions and support early warning systems.

2.Objectives
Classify regions into wildfire risk categories
Use weather and vegetation data for prediction
Compare multiple machine learning models
Visualize wildfire risk spatially
Evaluate model performance

3.Dataset
## Dataset Used

This project utilizes the NASA FIRMS (Fire Information for Resource Management System) dataset, which provides near real-time wildfire detection data collected from satellite observations. The dataset contains important wildfire-related parameters that help in analyzing and predicting wildfire risk levels.

### Features Used

The following features were used for training the Random Forest model:

- Latitude
- Longitude
- Brightness
- Scan
- Track
- Acquisition Date
- Acquisition Time
- Satellite
- Instrument
- Confidence
- Version
- Brightness Temperature (Bright_T31)
- Fire Radiative Power (FRP)
- Day/Night Indicator

Additional weather-related parameters were also incorporated for improving prediction accuracy.

### Target Variable

The target variable in this project is:

- **Wildfire Risk Level**

The wildfire risk was classified into categories such as:
- Low Risk
- Medium Risk
- High Risk



6.Methodology
1️⃣ Data Preprocessing
Handle missing values
Encode categorical variables
Normalize/scale features
Convert target into classification labels
2️⃣ Exploratory Data Analysis (EDA)
Distribution plots
Correlation heatmaps
Feature importance insights
3️⃣ Model Building

We implemented the following models:

Support Vector Machine (SVM)
Random Forest
XGBoost

4️⃣ Model Evaluation
Accuracy
Precision, Recall, F1-score
Confusion Matrix

5️⃣ Visualization
Risk level distribution
Feature importance plots
Spatial wildfire risk maps
🧠 Models Used
🔹 Support Vector Machine (SVM)
Effective for classification with clear margins
🔹 Random Forest
Ensemble method for better accuracy and robustness
🔹 XGBoost
Gradient boosting algorithm for high performance
📈 Results
Comparison of model performance
Best performing model identified based on evaluation metrics
Visualization of predicted wildfire risk
🗺️ Risk Mapping
Geographic visualization of wildfire-prone areas
Helps in understanding spatial distribution of risk

7.Tech Stack
Python
Pandas, NumPy
Scikit-learn
XGBoost
Matplotlib, Seaborn
(Optional) Folium / GeoPandas for maps


📂 Project Structure
├── data/
├── notebooks/
├── src/
│   ├── preprocessing.py
│   ├── models.py
│   ├── evaluation.py
├── outputs/
│   ├── plots/
│   ├── maps/
├── README.md
├── requirements.txt

 9.How to Run
# Clone the repository
git clone https://github.com/your-username/wildfire-risk-classification.git

# Navigate to project
cd wildfire-risk-classification

# Install dependencies
pip install -r requirements.txt

# Run the notebook or script
python src/models.py


10.Future Improvements
Use deep learning models (LSTM, CNN for spatial data)
Integrate real-time satellite data
Deploy as a web application
Improve spatial prediction using GIS tools

11.Contributors
Anagha Jayarajan
Akshay
Lekshmi Priya

This project is for academic and research purposes.
