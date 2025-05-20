# Model Training

# Load libraries
import pandas as pd 
import numpy as np 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import joblib
import os 

# Importing and reading datasets from data 
X_train = pd.read_csv("/Users/mandiraghimire/Desktop/MghiGitManu/machine-learning-introduction-Mandiraghi/data/X_train.csv")
y_train = pd.read_csv("/Users/mandiraghimire/Desktop/MghiGitManu/machine-learning-introduction-Mandiraghi/data/y_train.csv")

# Print the shape of training sets 
print("X_train Shape ")
print(X_train.shape)
print("y_train Shape")
print(y_train.shape)

# Linear Regression model train 
model = LinearRegression()
model.fit(X_train, y_train)

# Saving the Trained model 
os.makedirs("models", exist_ok= True)
joblib.dump(model, "models/linear_regression_model.pkl")

# Training Data Model Evaluation
y_train_pred = model.predict(X_train)
mse = mean_squared_error(y_train, y_train_pred)
r2 = r2_score(y_train, y_train_pred)

print("Training MSE: ")
print(mse)
print("Training R^2 Score :")
print(r2)

# Cross-Validation 
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
print("Cross-Validation R2 Scores: ")
print(cv_scores)
