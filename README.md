# Health Risk Prediction Project

## Requirements

**Software:**

- Python 3.11  
- Libraries: pandas, scikit-learn, xgboost, openpyxl, matplotlib, numpy

**Hardware:**

- Dell Latitude 7400 laptop  
- Intel(R) Core(TM) i7-8665U CPU @ 1.90GHz  
- 16 GB RAM

**Development Environment:**

- IntelliJ
- Windows 11 Pro (64-bit)

---

## Instructions to Run the Application

A. Clone the GitHub repository to your local machine:  
```bash
git clone https://github.com/cajhughes788/health_risk_prediction.git
cd health_risk_prediction
B. Open the project folder in your preferred IDE (VS Code or PyCharm).
C. Install required libraries
D. Ensure the input dataset DQN1 Dataset.xlsx is present in the root project directory.
E. Run algorithm.py to:

Load and preprocess the dataset

Build and train an XGBoost regression model

Apply early stopping and regularization

Optionally combine XGBoost with Random Forest using VotingRegressor

Output MSE and R² evaluation metrics before and after optimization

All output will be printed to the terminal upon running the script.

Note
Model training, optimization, and evaluation steps were implemented together in the algorithm.py script for simplicity. Hyperparameter tuning (learning rate, number of estimators), early stopping, and ensemble learning are all included and documented within the file.

Project Summary
This project predicts a healthRiskScore based on environmental and temporal features such as:

Temperature, humidity, wind speed/gust

PM2.5, NO2, CO2 levels

UV index, cloud cover, precipitation

Day of week, month, weekend indicator

Metric	Before Optimization	After Optimization
MSE	0.0177	0.0166
R²	0.9611	0.9634

This demonstrates that tuning and combining models can meaningfully improve predictive performance for real-world health applications.
