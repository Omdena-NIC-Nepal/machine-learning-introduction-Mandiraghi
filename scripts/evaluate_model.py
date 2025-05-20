# Model Evaluation

# Load libraries 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load Testing set of data from data folder 
X_test = pd.read_csv("/Users/mandiraghimire/Desktop/MghiGitManu/machine-learning-introduction-Mandiraghi/data/X_test.csv")
y_test = pd.read_csv("/Users/mandiraghimire/Desktop/MghiGitManu/machine-learning-introduction-Mandiraghi/data/y_test.csv")

# Load the saved model
model = joblib.load("models/linear_regression_model.pkl")

# Prediction on test data 
y_pred = model.predict(X_test)

# Evaluating the Regression metrics 
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mse)

print("Testing MSE: ")
print(mse)
print("Testing R^2 Score :")
print(r2)
print("Testing RMSE: ")
print(rmse)

# Flatten y_test and y_pred if needed
y_test = y_test.squeeze()
y_pred = y_pred.squeeze()

residuals = (y_test - y_pred)

# Plot the Residuals 
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(0, color='red', linestyle='--')
plt.title("Residuals vs Predicted")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.show()

# Plot Actual vs Predicted 
plt.figure(figsize=(6, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values")
plt.show()


