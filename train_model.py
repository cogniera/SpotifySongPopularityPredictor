#Author: Paarth Sharma 
#Filename: train model 
#Project Name: spotify popularity predictor 
#Creation Date: Nov 23 2025
#Modification Date: Nov 25 2025
#Description: imports the split test and training data, trains the data and also does Kfold validation to check if the model is overfitting, exports the model and stores it in a model folder 
import pandas as pd
import os
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

#paths for the processed data and model dumps
processed_data_path = "./data/processed"
model_path = "./models"

# Load processed data
X_train = pd.read_csv(f"{processed_data_path}/X_train.csv")
Y_train = pd.read_csv(f"{processed_data_path}/Y_train.csv").values.ravel()
X_val   = pd.read_csv(f"{processed_data_path}/X_val.csv")
Y_val   = pd.read_csv(f"{processed_data_path}/Y_val.csv").values.ravel()

#simple default GradientBoostingRegressor algorithm
base_model = GradientBoostingRegressor(random_state=42)

#use k fold on the algorithm to avoid over fitting 
kf = KFold(n_splits=5, shuffle=True, random_state=42)

#model evaluation scores 
cv_scores = []

print("Running 5-fold CV using default GradientBoostingRegressor...\n")

#loop through each fold 
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), start=1):
    
    #select the training and validation data from the entire data set for training for each fold
    X_tr, X_va = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_va = Y_train[train_idx], Y_train[val_idx]

    print(f"Training Fold {fold}")

    #training a  temp model on the training batch of the fold 
    model = GradientBoostingRegressor(random_state=42)
    
    #fit the temp model 
    model.fit(X_tr, y_tr)

    #get the predictions for the temporary model 
    preds = model.predict(X_va)

    #calculate the root mean squared error 
    rmse = mean_squared_error(y_va, preds, squared=False)

    print(f"Fold {fold} RMSE: {rmse:.4f}")
    #add the kfold scores to the cross validation scores array
    cv_scores.append(rmse)

#calculate the average mean squared error 
avg_rmse = sum(cv_scores) / len(cv_scores)
print(f"\nAverage CV RMSE: {avg_rmse:.4f}")

#final model on Train+Val
X_full = pd.concat([X_train, X_val], axis=0)
y_full = pd.concat([pd.Series(Y_train), pd.Series(Y_val)], axis=0).values

#load the final model 
final_model = GradientBoostingRegressor(random_state=42)

#train the final model 
final_model.fit(X_full, y_full)

#make the model dump directory 
os.makedirs(model_path, exist_ok=True)

#dump the final model as a pikle file 
joblib.dump(final_model, f"{model_path}/gbr_model.pkl")

print("\nFinal model trained and saved → models/gbr_model.pkl")
