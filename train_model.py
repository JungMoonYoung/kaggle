import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import os

# Create models directory if it doesn't exist
os.makedirs("../models", exist_ok=True)

# Load preprocessed data
train = pd.read_csv("../data/preprocessed_train.csv")

# Split data into training and validation sets
X = train.drop(columns=['sales'])
y = train['sales']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Make predictions with Linear Regression
lr_predictions = lr_model.predict(X_valid)

# Evaluate Linear Regression model
lr_mse = mean_squared_error(y_valid, lr_predictions)
lr_mae = mean_absolute_error(y_valid, lr_predictions)
print(f'Linear Regression - MSE: {lr_mse}, MAE: {lr_mae}')

# Save Linear Regression model
joblib.dump(lr_model, "../models/linear_regression_model.pkl")
print("Linear Regression model saved to '../models/linear_regression_model.pkl'")

# Train a Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions with Random Forest
rf_predictions = rf_model.predict(X_valid)

# Evaluate Random Forest model
rf_mse = mean_squared_error(y_valid, rf_predictions)
rf_mae = mean_absolute_error(y_valid, rf_predictions)
print(f'Random Forest - MSE: {rf_mse}, MAE: {rf_mae}')

# Save Random Forest model
joblib.dump(rf_model, "../models/random_forest_model.pkl")
print("Random Forest model saved to '../models/random_forest_model.pkl'")
