#Author:Paarth Sharma 
#Filename: data processing
#Project Name: spotify popularity predictor 
#Creation Date: Nov 20 2025
#Modification Date: Nov 25 2025
#Description: Preprocessed the spotify data, encodes categorical variables, scales the dataset, drops useless columns , splits the dataset into train, test and validation and dumps the model and datasets into seperate folders 
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder 
import os

raw_data_path = "./data/raw/spotify.csv"
processed_data_path = "./data/processed"
models_path = "./models"

df = pd.read_csv(raw_data_path)

#handle Duplicates
df = df.drop_duplicates(subset=["track_id"])

#handle Missing Values
df = df.dropna(subset=["popularity"])  
df = df.fillna(method="ffill").fillna(method="bfill")


#drop the invalid values from the data 
df = df[df["popularity"] > 4]

#select the target variable
target = "popularity"

#create a list of numerical variables 
numerical_features = [
    "duration_ms", "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo", "encoded_genre", "explicit"
]

#create a list of audio features 
audio_features = [col for col in numerical_features if col not in ("encoded_genre", "explicit")]

#filter out bad values 
df = df[~(df[audio_features] == 0).all(axis=1)]

#convert true and false into 0 and 1 
df["explicit"] = df["explicit"].astype(int)

#split data int training testing and validation 
X = df.drop(columns=[target])

#select only numerical columns 
X = X.select_dtypes(include=['number'])

#exclude the index columns 
X = X.loc[:, ~X.columns.str.contains("^Unnamed")]

#encode the genre variables as lable encoding 
lb_enc = LabelEncoder();
X['encoded_genre'] = lb_enc.fit_transform(df['track_genre'])

#select the target variables 
Y = df[target]

#split into training and other data 
X_train, X_temp, Y_train, Y_temp = train_test_split(
    X, Y, test_size=0.30, random_state=42
)

#split the other data into test and validation 
X_val, X_test, Y_val, Y_test = train_test_split(
    X_temp, Y_temp, test_size=0.50, random_state=42
)

#scale only numeric columns
scaler = StandardScaler()

#transform the numerical features by scaling them 
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_val[numerical_features] = scaler.transform(X_val[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])
X[numerical_features] = scaler.transform(X[numerical_features])

#Save processed data
os.makedirs(processed_data_path, exist_ok=True)
os.makedirs(models_path, exist_ok=True)

#save files to data 
X_train.to_csv(f"{processed_data_path}/X_train.csv", index=False)
Y_train.to_csv(f"{processed_data_path}/Y_train.csv", index=False)

X_val.to_csv(f"{processed_data_path}/X_val.csv", index=False)
Y_val.to_csv(f"{processed_data_path}/Y_val.csv", index=False)

X_test.to_csv(f"{processed_data_path}/X_test.csv", index=False)
Y_test.to_csv(f"{processed_data_path}/Y_test.csv", index=False)

X.to_csv(f"{processed_data_path}/numeric_raw_data.csv", index=False)

#dump models
joblib.dump(scaler, f"{processed_data_path}/scaler.pkl")
joblib.dump(scaler, f"{models_path}/scaler.pkl")
joblib.dump(lb_enc, f"{models_path}/label_encoder.pkl")


print("Preprocessing complete → data/processed/")
