# D682 - Task C1
# Predicting Health Risk Score using XGBoost Regressor
# This script loads the data, trains a model, and makes predictions

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset from the Excel file
df = pd.read_excel("DQN1 Dataset.xlsx")

# Task 1 - C1: Select the columns (features) we want the model to learn from
features = [
    'temp', 'humidity', 'windgust', 'windspeed',
    'pressure', 'cloudcover', 'pm2.5', 'no2', 'co2',
    'uvindex', 'precip', 'precipprob',
    'month', 'dayOfWeek', 'isWeekend'
]

# Task 1 - C1: This is the value we want the model to predict
target = 'healthRiskScore'

# Task 1 - C1: Make sure isWeekend is numeric
df['isWeekend'] = df['isWeekend'].astype(int)

# Task 1 - C1: Drop rows with missing data
df = df.dropna(subset=features + [target])

# Task 1 - C1: Prepare training and testing data
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Task 1 - C1: Create and train the model
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5)
model.fit(X_train, y_train)

# Task 1 - D3: Interpretation of evaluation metrics in report

# Task 2: B1 Update XGBoost model with basic optimization
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    learning_rate=0.05,        # Slower learning to improve accuracy
    n_estimators=200,          # More trees to allow deeper learning
    max_depth=5,
    early_stopping_rounds=10   # Stop training early if performance plateaus
)

# Task 2: B1 Train the optimized model using early stopping
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

# Task 2 - B2: Regularization
# Note: This model is used for ensemble so early stopping is removed
xgb_for_ensemble = xgb.XGBRegressor(
    objective='reg:squarederror',
    learning_rate=0.05,
    n_estimators=200,
    max_depth=5,
    reg_lambda=1.0,   # helps prevent overfitting
    reg_alpha=0.5     # makes the model simpler
)
xgb_for_ensemble.fit(X_train, y_train)

# Task 2 - B3: Apply two ensemble learning techniques
from sklearn.ensemble import RandomForestRegressor, VotingRegressor

# Train a basic Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Combine XGBoost and Random Forest using Voting Regressor
voting_model = VotingRegressor(estimators=[
    ('xgb', xgb_for_ensemble), ('rf', rf_model)
])
voting_model.fit(X_train, y_train)

# Task 2 - C1: Evaluate

# (originally Task 1 - C2) Predict on the test set using the voting model
y_pred = voting_model.predict(X_test)

# Evaluate the model using two metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the evaluation results
print("Evaluation Metrics for Task 2 - C1:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared Score (R²): {r2:.4f}")

# Task 2 - C2: completed model performance comparison in report