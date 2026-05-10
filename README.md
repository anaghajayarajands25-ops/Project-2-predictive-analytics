# Wildfire Risk Classification from Satellite and Weather Data

## Project Overview
This project focuses on classifying wildfire risk levels (Low, Medium, High) using satellite and meteorological data.  
Machine Learning models were trained to predict wildfire risk based on environmental and weather-related features.

The project uses the NASA FIRMS dataset for wildfire detection and analysis.

## Problem Statement
Classify regions into wildfire risk levels using meteorological features such as temperature, wind speed, humidity, drought index, and satellite-derived vegetation indices.

Apply the following Machine Learning models:
- Support Vector Machine (SVM)
- Random Forest
- XGBoost

Visualize wildfire risk maps and evaluate spatial prediction accuracy.

## Dataset
- Dataset: NASA FIRMS Dataset
- Source: NASA Fire Information for Resource Management System (FIRMS)

## Features Used
The following features were used for model training and prediction:

- Latitude
- Longitude
- Brightness
- Scan
- Track
- Confidence
- Brightness T31
- FRP (Fire Radiative Power)
- Day/Night
- Temperature
- Humidity
- Wind Speed
- Drought Index

## Target Variable
- Wildfire Risk Classification
  - Low Risk
  - Medium Risk
  - High Risk

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- Google Colab

## Machine Learning Models Used
- Support Vector Machine (SVM)
- Random Forest Classifier
- XGBoost Classifier

## Steps Performed
1. Data Collection
2. Data Preprocessing
3. Feature Engineering
4. Data Visualization
5. Train-Test Split
6. Model Training
7. Model Evaluation
8. Risk Prediction and Analysis

## Model Evaluation Metrics
The models were evaluated using:
- Accuracy Score
- Precision
- Recall
- F1-Score
- Confusion Matrix

## Project Outcome
The project successfully classified wildfire risk levels using machine learning techniques and satellite data analysis.

## Files Included
- Google Colab Notebook
- README.md

## How to Run
1. Open the notebook in Google Colab.
2. Upload the dataset.
3. Run all cells sequentially.
4. View the predictions, graphs, and evaluation metrics.

## Future Improvements
- Real-time wildfire monitoring
- Deployment using Streamlit or Flask
- Integration with live weather APIs
- Improved spatial visualization techniques

## Contributors
- Anagha Jayarajan
- Lekshmi Priya
- Akshay
