
# ============================================================
# PROJECT: POLICY RENEWAL CHURN PREDICTION
# FILE: eda_large.py
# PURPOSE: Exploratory Data Analysis on Large Dataset
# ============================================================


# ============================================================
# SECTION 1: IMPORT LIBRARIES
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# SECTION 2: LOAD LARGE DATASET
# ============================================================

# Load the 5000-customer dataset
df = pd.read_csv("data/insurance_customers_large.csv")

print("\nLarge dataset loaded successfully!")


# ============================================================
# SECTION 3: DATASET OVERVIEW
# ============================================================

# Display total rows and columns
print("\nDataset shape:", df.shape)

# Display column names, data types and missing values
print("\nDataset information:")
df.info()

# Display statistical summary
print("\nStatistical summary:")
print(df.describe())


# ============================================================
# SECTION 4: CHECK MISSING VALUES
# ============================================================

# Count missing values in each column

print("\nMissing values:")
print(df.isnull().sum())

# Calculate total missing values
print("\nTotal missing values:", df.isnull().sum().sum())


# ============================================================
# SECTION 5: CHECK DUPLICATE RECORDS
# ============================================================

# Identify duplicate customer records

print("\nDuplicate records:")
print(df.duplicated().sum())


# ============================================================
# SECTION 6: CHURN DISTRIBUTION
# ============================================================

# Count renewed and churned customers
# 0 = Renewed
# 1 = Churned

print("\nChurn distribution:")
print(df["churn"].value_counts())

# Calculate churn percentage

print("\nChurn percentage:")
print(df["churn"].value_counts(normalize=True) * 100)


# ============================================================
# SECTION 7: CHURN DISTRIBUTION VISUALIZATION
# ============================================================

# Create a bar chart for renewed vs churned customers

df["churn"].value_counts().sort_index().plot(kind="bar")

plt.title("Customer Churn Distribution - Large Dataset")
plt.xlabel("Churn (0 = Renewed, 1 = Churned)")
plt.ylabel("Number of Customers")
plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig("data/large_churn_distribution.png")

print("\nChurn distribution chart saved successfully!")


# ============================================================
# SECTION 8: NUMERICAL ANALYSIS BY CHURN
# ============================================================

# Compare average customer characteristics
# between renewed and churned customers

print("\nAverage customer details by churn category:")

print(
    df.groupby("churn")[
        ["age", "premium", "policy_tenure", "complaints", "payment_delay"]
    ].mean()
)


# ============================================================
# SECTION 9: CORRELATION ANALYSIS
# ============================================================

# Calculate correlation between numerical features

correlation = df.corr(numeric_only=True)

print("\nCorrelation matrix:")
print(correlation)


# ============================================================
# SECTION 10: CORRELATION VISUALIZATION
# ============================================================

# Display correlation matrix as a heatmap

plt.figure(figsize=(9, 7))

plt.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)

plt.colorbar(label="Correlation")

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(correlation.columns)),
    correlation.columns
)

plt.title("Feature Correlation Matrix - Large Dataset")

plt.tight_layout()

plt.savefig("data/large_correlation_matrix.png")

print("\nCorrelation matrix saved successfully!")


# ============================================================
# SECTION 11: COMPLETION
# ============================================================

print("\n============================================================")
print("LARGE DATASET EDA COMPLETED SUCCESSFULLY!")
print("============================================================")
