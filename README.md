# Spotify Song Popularity Predictor  
Author: Paarth Sharma  

A complete end-to-end machine learning project that predicts the popularity score of a Spotify track using structured audio features and encoded genre information. This repository includes preprocessing scripts, model training, evaluation, and an interactive Streamlit app that ties everything together.
Dataset used : https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset
---

## Overview

This project explores whether it is possible to predict Spotify track popularity using only numeric features such as danceability, tempo, valence, loudness, and other audio indicators. The workflow covers data cleaning, encoding, scaling, model training, cross-validation, evaluation, and deployment.

The final result is a working ML pipeline with a user-friendly UI that demonstrates predictions, dataset insights, model performance, and feature importances.

---

## Features

- Full preprocessing pipeline (cleaning, filtering, encoding, scaling, splitting)
- Genre encoding using LabelEncoder
- Standardization of all numeric features
- GradientBoostingRegressor with 5-fold cross-validation
- Separate train/validation/test splits
- Streamlit application with:
  - Popularity prediction
  - Dataset preview
  - Heatmaps and scatter plots
  - Feature importance visualization
  - Actual vs predicted comparison

---

## Tech Stack

- Python  
- pandas  
- numpy  
- scikit-learn (LabelEncoder, StandardScaler, GradientBoostingRegressor, KFold)  
- Streamlit  
- Plotly  
- joblib  

---

## Pipeline Breakdown

### 1. Preprocessing (`data_processing.py`)
- Drops duplicates and missing popularity rows  
- Forward/backward fills remaining missing values  
- Removes rows where all audio features are zero  
- Converts `explicit` into 0/1  
- Encodes `track_genre`  
- Selects numeric columns  
- Scales all features  
- Splits data into train, validation, and test  
- Saves processed datasets and transformation models

### 2. Training (`train_model.py`)
- Loads processed datasets  
- Runs 5-fold cross-validation  
- Computes RMSE per fold  
- Trains final model on full training + validation  
- Saves final model to `models/`

### 3. Evaluation (`evaluate_model.py`)
- Loads the trained model  
- Evaluates performance on the test set  
- Prints RMSE, MAE, and R²

### 4. Streamlit Application (`app.py`)
- Loads the trained model, scaler, and label encoder  
- User inputs track features in a clean UI  
- Encodes genre on the fly  
- Predicts popularity  
- Displays statistics, heatmaps, scatter plots, and feature importances

---

## Model Performance

Popularity is extremely difficult to predict from audio features alone. Many real-world factors are not in the dataset (artist popularity, playlist placement, promotion, viral trends).  

Given that, the model’s performance is within expected ranges:

- RMSE: around 14  
- MAE: around 11  
- R²: roughly 0.35–0.40  

The goal of this project is not perfect accuracy but building a full ML pipeline and deployment.

---

## Running the Project

### for running this project open Docker on your Device and then run the docker file using the following : 
docker build -t spotify-ml .    


docker run -p 8501:8501 spotify-ml-app
