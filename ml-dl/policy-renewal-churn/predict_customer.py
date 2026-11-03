
# ============================================================
# PROJECT: POLICY RENEWAL CHURN PREDICTION
# FILE: predict_customer.py
# PURPOSE: Predict churn for a new insurance customer
# ============================================================


# ============================================================
# 1. IMPORT REQUIRED LIBRARIES
# ============================================================

import joblib
import pandas as pd

from pathlib import Path


# ============================================================
# 2. DEFINE PROJECT PATHS
# ============================================================

# Get the project directory
BASE_DIR = Path(__file__).resolve().parent

# Saved trained model
MODEL_PATH = (
    BASE_DIR / "models" / "logistic_regression_pipeline.joblib"
)


# ============================================================
# 3. LOAD TRAINED MODEL
# ============================================================

print("\n" + "=" * 60)
print("INSURANCE CUSTOMER CHURN PREDICTION")
print("=" * 60)

# Check whether the trained model exists
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found: {MODEL_PATH}. "
        "Run logistic_regression.py first."
    )

# Load the complete pipeline (scaler + classifier)
model = joblib.load(MODEL_PATH)

print("\nTrained Logistic Regression model loaded successfully!")


# ============================================================
# 4. GET CUSTOMER INFORMATION
# ============================================================

print("\nEnter the customer details below:")

age = int(input("Customer age: "))

premium = float(input("Annual premium: "))

policy_tenure = int(input("Policy tenure (years): "))

complaints = int(input("Number of complaints: "))

payment_delay = int(input("Payment delay (days): "))


# ============================================================
# 5. VALIDATE CUSTOMER INPUT
# ============================================================

# Basic validation to prevent invalid input values

if not 18 <= age <= 75:
    raise ValueError("Age must be between 18 and 75.")

if premium <= 0:
    raise ValueError("Premium must be greater than zero.")

if not 1 <= policy_tenure <= 20:
    raise ValueError("Policy tenure must be between 1 and 20.")

if complaints < 0:
    raise ValueError("Complaints cannot be negative.")

if not 0 <= payment_delay <= 90:
    raise ValueError("Payment delay must be between 0 and 90 days.")


# ============================================================
# 6. CONVERT INPUT INTO MODEL FORMAT
# ============================================================

# The feature names and order must match training data.

customer_data = pd.DataFrame([{
    "age": age,
    "premium": premium,
    "policy_tenure": policy_tenure,
    "complaints": complaints,
    "payment_delay": payment_delay
}])

print("\nCustomer information received successfully!")


# ============================================================
# 7. MAKE CHURN PREDICTION
# ============================================================

# Predict class:
# 0 = Renewed
# 1 = Churned

prediction = model.predict(customer_data)[0]

# Get probability of churn (class 1)
churn_probability = model.predict_proba(
    customer_data
)[0][1]

# Convert probability into percentage
churn_percentage = churn_probability * 100


# ============================================================
# 8. ASSIGN RISK CATEGORY
# ============================================================

# These are initial illustrative probability bands.
# They are not calibrated business decision thresholds.

if churn_probability < 0.30:
    risk_category = "LOW"

elif churn_probability < 0.60:
    risk_category = "MEDIUM"

else:
    risk_category = "HIGH"


# ============================================================
# 9. DISPLAY PREDICTION RESULTS
# ============================================================

print("\n" + "=" * 60)
print("CUSTOMER CHURN PREDICTION RESULT")
print("=" * 60)

print("\nCustomer Details:")
print(customer_data.to_string(index=False))

print("\nPrediction:")

if prediction == 1:
    print("Customer is predicted to CHURN.")
else:
    print("Customer is predicted to RENEW.")

print(f"\nChurn Probability: {churn_percentage:.2f}%")

print(f"Risk Category: {risk_category}")

print("\n" + "=" * 60)
print("PREDICTION COMPLETED SUCCESSFULLY!")
print("=" * 60)
