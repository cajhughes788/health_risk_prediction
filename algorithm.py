# D682 - Task C1
# Predicting Health Risk Score using XGBoost Regressor
# This script loads the data, trains a model, and makes predictions

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Load the dataset from the Excel file
df = pd.read_excel("DQN1 Dataset.xlsx")

# Select the columns (features) we want the model to learn from
features = [
    'temp', 'humidity', 'windgust', 'windspeed',
    'pressure', 'cloudcover', 'pm2.5', 'no2', 'co2',
    'uvindex', 'precip', 'precipprob',
    'month', 'dayOfWeek', 'isWeekend'
]

# This is the value we want the model to predict
target = 'healthRiskScore'

# Make sure isWeekend is numeric
df['isWeekend'] = df['isWeekend'].astype(int)

# Drop rows with missing data
df = df.dropna(subset=features + [target])

# Prepare training and testing data
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

from sklearn.metrics import mean_squared_error, r2_score

# Evaluate the model using two metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the evaluation results
print("Evaluation Metrics for Task D2:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared Score (R²): {r2:.4f}")
