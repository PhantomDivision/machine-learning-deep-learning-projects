
# ============================================================
# PROJECT: POLICY RENEWAL CHURN PREDICTION
# FILE: create_large_dataset.py
# PURPOSE: Generate a large synthetic insurance dataset
# ============================================================


# ============================================================
# SECTION 1: IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np


# ============================================================
# SECTION 2: CONFIGURATION
# ============================================================

# Number of customers to generate
NUM_CUSTOMERS = 5000

# Random seed ensures reproducible results
np.random.seed(42)


# ============================================================
# SECTION 3: GENERATE CUSTOMER FEATURES
# ============================================================

# Generate customer ages between 18 and 75
age = np.random.randint(18, 76, NUM_CUSTOMERS)

# Generate annual insurance premiums
premium = np.random.randint(3000, 25001, NUM_CUSTOMERS)

# Generate policy tenure in years
policy_tenure = np.random.randint(1, 21, NUM_CUSTOMERS)

# Generate number of customer complaints
complaints = np.random.poisson(lam=1.5, size=NUM_CUSTOMERS)

# Generate payment delays in days
payment_delay = np.random.exponential(
    scale=12,
    size=NUM_CUSTOMERS
).astype(int)

# Limit payment delays to a maximum of 90 days
payment_delay = np.clip(payment_delay, 0, 90)


# ============================================================
# SECTION 4: GENERATE CHURN TARGET
# ============================================================

# Create a churn risk score using customer characteristics.
# Higher complaints and payment delays increase churn risk.
# Longer tenure and older age reduce churn risk in this
# synthetic teaching example.

risk_score = (
    -3.0
    + 0.035 * (40 - age)
    + 0.12 * complaints
    + 0.025 * payment_delay
    - 0.06 * policy_tenure
)

# Convert risk score into a probability between 0 and 1
churn_probability = 1 / (1 + np.exp(-risk_score))

# Generate churn values:
# 0 = Renewed
# 1 = Churned
churn = np.random.binomial(
    n=1,
    p=churn_probability,
    size=NUM_CUSTOMERS
)


# ============================================================
# SECTION 5: CREATE DATAFRAME
# ============================================================

# Combine all generated features and target into a DataFrame

df = pd.DataFrame({
    "age": age,
    "premium": premium,
    "policy_tenure": policy_tenure,
    "complaints": complaints,
    "payment_delay": payment_delay,
    "churn": churn
})


# ============================================================
# SECTION 6: SAVE DATASET
# ============================================================

# Save the large dataset separately.
# The original 10-customer CSV will not be overwritten.

df.to_csv(
    "data/insurance_customers_large.csv",
    index=False
)


# ============================================================
# SECTION 7: DISPLAY DATASET INFORMATION
# ============================================================

print("\nLarge dataset created successfully!")

print("\nTotal customers:", len(df))

print("\nDataset shape:", df.shape)

print("\nFirst 5 customer records:")
print(df.head())

print("\nChurn distribution:")
print(df["churn"].value_counts())

print("\nChurn percentage:")
print(df["churn"].value_counts(normalize=True) * 100)

print("\nDataset saved at:")
print("data/insurance_customers_large.csv")


# ============================================================
# SECTION 8: COMPLETION
# ============================================================

print("\n============================================================")
print("LARGE DATASET GENERATION COMPLETED!")
print("============================================================")
