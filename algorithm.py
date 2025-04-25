# D682 - Task C1
# Predicting Health Risk Score using XGBoost Regressor
# This script loads the data, trains a model, and evaluates how well it predicts the healthRiskScore

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset from the Excel file
df = pd.read_excel("DQN1 Dataset.xlsx")

# Select the columns (features) we want the model to learn from
# These include weather and pollution-related factors
features = [
    'temp', 'humidity', 'windgust', 'windspeed',
    'pressure', 'cloudcover', 'pm2.5', 'no2', 'co2',
    'uvindex', 'precip', 'precipprob',
    'month', 'dayOfWeek', 'isWeekend'
]

# This is the value we want the model to predict
target = 'healthRiskScore'

# isWeekend is in the right format (0 or 1 instead of True/False)
df['isWeekend'] = df['isWeekend'].astype(int)

# Get rid of any rows that have missing values so they don't mess up the training
df = df.dropna(subset=features + [target])

# Split the data into inputs (X) and output (y)
X = df[features]
y = df[target]

# Split the data into training and testing sets
# 80% will be used to train the model, and 20% to test it
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the XGBoost model
# 'reg:squarederror' means we’re doing regression, not classification
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5)

# Train (fit) the model on the training data
model.fit(X_train, y_train)

# Now let’s see how it does by making predictions on the test set
y_pred = model.predict(X_test)

# Check how well the model did using two metrics:
# Mean Squared Error (MSE) – lower is better
# R-squared (R²) – closer to 1 is better
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the results
print("Model Evaluation Results:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")
