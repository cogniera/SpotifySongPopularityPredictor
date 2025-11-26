#Author: Paarth Sharma 
#Filename: evaluate model 
#Project Name: spotify popularity predictor 
#Creation Date: Nov 23 2025
#Modification Date: Nov 25 2025 
#Description: Reads in the testing dataset , imports the trained model and tests it on the testing split of the dataset , gives metrics like root mean squared error, Mean absolute error , and r^2 score 
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

#paths for the data and model 
processed_data_path = "./data/processed"
model_path = "./models/gbr_model.pkl"

#load data 
X_test = pd.read_csv(f"{processed_data_path}/X_test.csv")
Y_test = pd.read_csv(f"{processed_data_path}/Y_test.csv").values.ravel()

#load the model 
model = joblib.load(model_path)

#predict using the model 
preds = model.predict(X_test)

#predict the matrics
rmse = mean_squared_error(Y_test, preds, squared=False)
mae = mean_absolute_error(Y_test, preds)
r2 = r2_score(Y_test, preds)

#print the matrics 
print("\nModel Evaluation on Test Set:")
print(f"RMSE: {rmse:.4f}")
print(f"MAE : {mae:.4f}")
print(f"R²  : {r2:.4f}")