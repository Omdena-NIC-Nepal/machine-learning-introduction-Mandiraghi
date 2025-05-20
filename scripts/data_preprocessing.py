# Data Preprocessing

# Load Libraries 
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# Load Dataset 
boston_housing = pd.read_csv("/Users/mandiraghimire/Desktop/MghiGitManu/machine-learning-introduction-Mandiraghi/data/BostonHousing.csv")

# Handle missing values
print("Boston Housing Missing VALUES:")
print(boston_housing.isnull().sum())

# Handle outliers
def remove_outliers_iqr(boston_housing, column):
    Q1 = boston_housing[column].quantile(0.25)
    Q3 = boston_housing[column].quantile(0.75)
    IQR = Q3 - Q1 
    lower = Q1 - 1.5 * IQR 
    upper = Q3 + 1.5 * IQR
    return boston_housing[(boston_housing[column] >= lower) & (boston_housing[column] <= upper)]

for col in boston_housing.columns: 
    boston_housing = remove_outliers_iqr(boston_housing, col)

print("Outliers removed")

# Encoding categorical variable 'chas'
boston_housing['chas'] = boston_housing['chas'].astype('category')
boston_housing = pd.get_dummies(boston_housing, columns=['chas'], drop_first=True)

print("Categorical encoding done")

# Define features and target variables
X = boston_housing.drop('medv', axis=1)
y = boston_housing['medv']

# Standardize numerical features 
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

print("Features scaled")

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("X_train shape:")
print(X_train.shape)
print("X_test shape:")
print(X_test.shape)

# Save Train and Test data to CSV
X_train.to_csv("/Users/mandiraghimire/Desktop/MghiGitManu/machine-learning-introduction-Mandiraghi/data/X_train.csv", index=False)
X_test.to_csv("/Users/mandiraghimire/Desktop/MghiGitManu/machine-learning-introduction-Mandiraghi/data/X_test.csv", index=False)
y_train.to_csv("/Users/mandiraghimire/Desktop/MghiGitManu/machine-learning-introduction-Mandiraghi/data/y_train.csv", index=False)
y_test.to_csv("/Users/mandiraghimire/Desktop/MghiGitManu/machine-learning-introduction-Mandiraghi/data/y_test.csv", index=False)
