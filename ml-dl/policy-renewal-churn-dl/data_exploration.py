
"""
Data Exploration
Project: Policy Renewal Churn Prediction - Deep Learning

Purpose:
Understand the insurance customer dataset before building
an Artificial Neural Network (ANN).
"""

import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

DATA_PATH = "data/insurance_customers_large.csv"
PLOT_DIR = "data/plots"

os.makedirs(PLOT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")


# --------------------------------------------------
# 2. Load Dataset
# --------------------------------------------------

def load_dataset():
    """Load the insurance customer dataset."""

    print("\n========== LOADING DATASET ==========")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print("Dataset loaded successfully!")
    return df


# --------------------------------------------------
# 3. Basic Dataset Information
# --------------------------------------------------

def inspect_dataset(df):

    print("\n========== DATASET SHAPE ==========")
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")

    print("\n========== FIRST 5 RECORDS ==========")
    print(df.head())

    print("\n========== COLUMN INFORMATION ==========")
    df.info()

    print("\n========== STATISTICAL SUMMARY ==========")
    print(df.describe())

    print("\n========== MISSING VALUES ==========")
    print(df.isnull().sum())

    print("\n========== DUPLICATE RECORDS ==========")
    print(f"Duplicate rows: {df.duplicated().sum()}")


# --------------------------------------------------
# 4. Target Variable Analysis
# --------------------------------------------------

def analyze_target(df):

    print("\n========== CHURN DISTRIBUTION ==========")

    churn_counts = df["churn"].value_counts().sort_index()
    churn_percentages = (
        df["churn"].value_counts(normalize=True).sort_index() * 100
    )

    print("Churn counts:")
    print(churn_counts)

    print("\nChurn percentages:")
    print(churn_percentages.round(2))

    print("\nTarget interpretation:")
    print("0 = Customer renews policy")
    print("1 = Customer churns / policy lapses")

    plt.figure(figsize=(7, 5))

    sns.countplot(
        data=df,
        x="churn",
        hue="churn",
        palette="Set2",
        legend=False
    )

    plt.title("Customer Churn Distribution")
    plt.xlabel("Churn (0 = Renewed, 1 = Churned)")
    plt.ylabel("Number of Customers")
    plt.tight_layout()

    plt.savefig(
        f"{PLOT_DIR}/churn_distribution.png",
        dpi=300
    )

    plt.close()


# --------------------------------------------------
# 5. Feature Distribution Analysis
# --------------------------------------------------

def analyze_feature_distributions(df):

    print("\n========== FEATURE DISTRIBUTIONS ==========")

    numerical_columns = [
        "age",
        "premium",
        "policy_tenure",
        "complaints",
        "payment_delay"
    ]

    for column in numerical_columns:

        plt.figure(figsize=(7, 5))

        sns.histplot(
            data=df,
            x=column,
            kde=True,
            bins=30,
            color="steelblue"
        )

        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")

        plt.tight_layout()

        plt.savefig(
            f"{PLOT_DIR}/{column}_distribution.png",
            dpi=300
        )

        plt.close()

        print(f"Saved distribution plot: {column}")


# --------------------------------------------------
# 6. Feature Comparison by Churn
# --------------------------------------------------

def analyze_features_by_churn(df):

    print("\n========== FEATURE ANALYSIS BY CHURN ==========")

    numerical_columns = [
        "age",
        "premium",
        "policy_tenure",
        "complaints",
        "payment_delay"
    ]

    print("\nAverage feature values by churn:")
    print(df.groupby("churn")[numerical_columns].mean().round(2))

    for column in numerical_columns:

        plt.figure(figsize=(7, 5))

        sns.boxplot(
            data=df,
            x="churn",
            y=column,
            hue="churn",
            palette="Set2",
            legend=False
        )

        plt.title(f"{column} vs Churn")
        plt.xlabel("Churn (0 = Renewed, 1 = Churned)")
        plt.ylabel(column)

        plt.tight_layout()

        plt.savefig(
            f"{PLOT_DIR}/{column}_vs_churn.png",
            dpi=300
        )

        plt.close()


# --------------------------------------------------
# 7. Correlation Analysis
# --------------------------------------------------

def analyze_correlations(df):

    print("\n========== CORRELATION MATRIX ==========")

    correlation_matrix = df.corr(numeric_only=True)

    print(correlation_matrix.round(3))

    plt.figure(figsize=(9, 7))

    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5
    )

    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()

    plt.savefig(
        f"{PLOT_DIR}/correlation_heatmap.png",
        dpi=300
    )

    plt.close()


# --------------------------------------------------
# 8. Main Execution
# --------------------------------------------------

def main():

    print("\n========================================")
    print(" POLICY RENEWAL CHURN - DATA EXPLORATION")
    print("========================================")

    df = load_dataset()

    inspect_dataset(df)

    analyze_target(df)

    analyze_feature_distributions(df)

    analyze_features_by_churn(df)

    analyze_correlations(df)

    print("\n========== EXPLORATION COMPLETED ==========")
    print(f"All plots saved inside: {PLOT_DIR}/")


if __name__ == "__main__":
    main()
