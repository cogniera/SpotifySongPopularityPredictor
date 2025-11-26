#Author: Paarth Sharma 
#Filename: app 
#Project Name: spotify popularity predictor 
#Creation Date: Nov 20 2025
#Modification Date: Nov 25 2025
#Description: Creates a stream lit app to predict the popularity of a new song based on it's features, plots the correlation between different features of the graph , heatmap of the popularity score and other features , shows the important features of the model. Shows different metrics of the model's performance to display the user 
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


st.set_page_config(page_title="Spotify Popularity Predictor", layout="wide")

#load data model 
prcessed_data = "./data/processed"
gbr_model = "./models/gbr_model.pkl"
label_encoder = "./models/label_encoder.pkl"
raw_data = "./data/raw/spotify.csv"
numeric_data = "./data/processed/numeric_raw_data.csv"

#store the cache for train and test data for displaying results 
@st.cache_data
def load_processed():
    X_train = pd.read_csv(f"{prcessed_data}/X_train.csv")
    Y_train = pd.read_csv(f"{prcessed_data}/Y_train.csv").values.ravel()
    X_val = pd.read_csv(f"{prcessed_data}/X_val.csv")
    Y_val = pd.read_csv(f"{prcessed_data}/Y_val.csv").values.ravel()
    X_test = pd.read_csv(f"{prcessed_data}/X_test.csv")
    Y_test = pd.read_csv(f"{prcessed_data}/Y_test.csv").values.ravel()
    return X_train, Y_train, X_val, Y_val, X_test, Y_test

#caching the model 
@st.cache_resource
def load_model(model_name):
    return joblib.load(model_name)

#caching raw data for display 
@st.cache_data
def load_raw(fileName):
    return pd.read_csv(fileName)

# Load train and test data 
X_train, Y_train, X_val, Y_val, X_test, Y_test = load_processed()

#load the gbr model 
gbr = load_model(gbr_model)

#load the encoder model 
lb_enc = load_model(label_encoder)

#load the raw data 
data_frame_raw = load_raw(raw_data)

#numeric raw data 
numeric_df = load_raw(numeric_data)

#extract the unique genre values 
unique_genres = sorted(data_frame_raw['track_genre'].unique())

#display the title 
st.title("Spotify Song Popularity Predictor")

#display the description for the App
st.write("Machine Learning project using GradientBoostingRegressor, full preprocessing, and evaluation dashboard.")

#display different tabs for displaying EDA and plots 
tabs = st.tabs(["Prediction", "Model Info", "Dataset", "Plots"])


# Prediction Tab
with tabs[0]:

    #display the header of the prediction tab
    st.header("Predict Popularity")

    #declare the colomns that are numerical
    numeric_cols = [
        "duration_ms", "danceability", "energy", "loudness",
        "speechiness", "acousticness", "instrumentalness",
        "liveness", "valence", "tempo", "explicit", "encoded_genre"
    ]

    #declare user input dictionary 
    user_input = {}

    #declare the subheader 
    st.subheader("Song Features")

    #create 2 columns 
    cols = st.columns(2)

    #loop through the numeric cols 
    for i, inputs in enumerate(numeric_cols):
        
        if inputs == "explicit" : continue

        if inputs == "encoded_genre" : continue

        #go through each column left to right 
        with cols[i % 2]:

            #creating labels and inputs for the features  
            user_input[inputs] = st.number_input(
                inputs, value=float(X_train[inputs].mean())
            )

    
    # Build feature row with all the columns as 0
    input_row = {col: 0 for col in X_train.columns}

    #create a selection for explicit 
    explicit = st.selectbox("Explicit", ["No", "Yes"])

    #put the user input for explicit in the user input dataframe 
    user_input["explicit"] = 1 if explicit == "Yes" else 0

    #create a selection for unique genre values 
    selected_genre = st.selectbox("Genre", unique_genres)

    #encode the selected value and then save that as input 
    encoded_genre = lb_enc.transform([selected_genre])[0]

    #put that into user input 
    user_input["encoded_genre"] = encoded_genre

    #for each column, the input row should be filled with the user input 
    for col in numeric_cols :
        input_row[col] = user_input[col]

    #create a pandas dataframe from the input 
    input_df = pd.DataFrame([input_row])

    #create the button and if it is pressed predict using the model 
    if st.button("Predict"):
        prediction = gbr.predict(input_df)[0]
        st.success(f"Predicted Popularity: **{prediction:.2f}**")

# Model Info Tab
with tabs[1]:

    #create header 
    st.header("Model Information")

    #create subheadings 
    st.subheader("GradientBoostingRegressor Parameters")

    #get the parameters for the model 
    st.json(gbr.get_params())

    #get the test predictions 
    preds = gbr.predict(X_test)

    #statistics from the prediction 
    rmse = mean_squared_error(Y_test, preds, squared=False)
    mae = mean_absolute_error(Y_test, preds)
    r2 = r2_score(Y_test, preds)

    #display the test performance 
    st.subheader("Test Set Performance")
    st.metric("RMSE", f"{rmse:.4f}")
    st.metric("MAE", f"{mae:.4f}")
    st.metric("R² Score", f"{r2:.4f}")

    #feature importance 
    st.subheader("Feature Importances")

    fi = pd.DataFrame({
        "feature": X_train.columns,
        "importance": gbr.feature_importances_
    }).sort_values("importance", ascending=False)

    fig = px.bar(fi, x="importance", y="feature", orientation="h")
    st.plotly_chart(fig, use_container_width=True)

#dataset Tab
with tabs[2]:
    #header 
    st.header("Dataset Preview")

    #display the initial few columns of the dataset 
    st.dataframe(data_frame_raw.head(50), use_container_width=True)

    #create a subheader 
    st.subheader("Basic Stats")

    #show the description of the data 
    st.write(data_frame_raw.describe())

#plots Tab
with tabs[3]:

    #create a header for the EDA 
    st.header("Exploratory Data Analysis")

    #popularity distribution
    fig1 = px.histogram(data_frame_raw, x="popularity", nbins=40, title="Popularity Distribution")
    st.plotly_chart(fig1, use_container_width=True)

    #correlation heatmap
    corr = numeric_df.corr()

    fig2 = px.imshow(
        corr,
        text_auto=False,
        title="Correlation Heatmap",
        aspect="auto"
    )
    st.plotly_chart(fig2, use_container_width=True)

    #danceability vs popularity
    fig3 = px.scatter(
        data_frame_raw,
        x="danceability", y="popularity",
        title="Danceability vs Popularity",
        opacity=0.4
    )
    st.plotly_chart(fig3, use_container_width=True)

    #actual vs predicted 
    pred_df = pd.DataFrame({"actual": Y_test, "pred": gbr.predict(X_test)})
    fig4 = px.scatter(pred_df, x="actual", y="pred", title="Actual vs Predicted Popularity")

    st.plotly_chart(fig4, use_container_width=True)
