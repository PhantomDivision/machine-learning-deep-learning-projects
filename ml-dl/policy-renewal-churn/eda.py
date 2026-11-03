
# ============================================================
# PROJECT: POLICY RENEWAL CHURN PREDICTION
# FILE: eda.py
# PURPOSE: Exploratory Data Analysis (EDA)
# ============================================================


# ============================================================
# SECTION 1: IMPORT LIBRARIES
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# SECTION 2: LOAD DATASET
# ============================================================

# Load the CSV dataset created by create_dataset.py
df = pd.read_csv("data/insurance_customers.csv")

print("\nDataset loaded successfully!")


# ============================================================
# SECTION 3: DATASET OVERVIEW
# ============================================================

# Display the number of rows and columns
print("\nDataset shape:", df.shape)

# Display column names, data types and non-null values
print("\nDataset information:")
df.info()

# Display statistical summary of numerical columns
print("\nStatistical summary:")
print(df.describe())


# ============================================================
# SECTION 4: CHURN DISTRIBUTION
# ============================================================

# Count customers in each churn category
# 0 = Renewed
# 1 = Churned

print("\nChurn distribution:")
print(df["churn"].value_counts())

# Calculate the percentage of each churn category
print("\nChurn percentage:")
print(df["churn"].value_counts(normalize=True) * 100)


# ============================================================
# SECTION 5: CHURN DISTRIBUTION VISUALIZATION
# ============================================================

# Create a bar chart showing renewed vs churned customers

df["churn"].value_counts().sort_index().plot(kind="bar")

plt.title("Customer Churn Distribution")
plt.xlabel("Churn (0 = Renewed, 1 = Churned)")
plt.ylabel("Number of Customers")
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig("data/churn_distribution.png")

print("\nChurn distribution chart saved to data/churn_distribution.png")


# ============================================================
# SECTION 6: NUMERICAL ANALYSIS BY CHURN
# ============================================================

# Compare average customer characteristics
# between renewed and churned customers.

print("\nAverage customer details by churn category:")

print(
    df.groupby("churn")[
        ["age", "premium", "policy_tenure", "complaints", "payment_delay"]
    ].mean()
)


# ============================================================
# SECTION 7: CORRELATION ANALYSIS
# ============================================================

# Calculate relationships between numerical columns.
# Correlation ranges from -1 to +1.

print("\nCorrelation matrix:")

correlation = df.corr(numeric_only=True)

print(correlation)

# Visualize the correlation matrix

plt.figure(figsize=(8, 6))

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

plt.title("Feature Correlation Matrix")

plt.tight_layout()
plt.savefig("data/correlation_matrix.png")

print("\nCorrelation matrix saved to data/correlation_matrix.png")


# ============================================================
# SECTION 8: COMPLETION
# ============================================================

print("\n============================================================")
print("EXPLORATORY DATA ANALYSIS COMPLETED SUCCESSFULLY!")
print("============================================================")
