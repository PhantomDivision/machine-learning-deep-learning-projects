
# ============================================================
# POLICY RENEWAL CHURN PREDICTION
# SHAP EXPLAINABILITY - LOGISTIC REGRESSION
# ============================================================


# =========================
# 1. IMPORT LIBRARIES
# =========================

import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

from pathlib import Path


# =========================
# 2. DEFINE PROJECT PATHS
# =========================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "insurance_customers_large.csv"

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "logistic_regression_tuned_pipeline.joblib"
)

# Folder to save SHAP visualizations
OUTPUT_DIR = BASE_DIR / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# 3. LOAD DATASET
# =========================

print("\nLoading insurance dataset...")

df = pd.read_csv(DATA_PATH)

# Separate input features and target
X = df.drop("churn", axis=1)

print("Dataset loaded:", df.shape)


# =========================
# 4. LOAD TUNED MODEL
# =========================

print("\nLoading tuned Logistic Regression model...")

model = joblib.load(MODEL_PATH)

print("Tuned model loaded successfully!")


# =========================
# 5. SELECT BACKGROUND DATA
# =========================

# SHAP needs representative background data
# to establish a baseline prediction.

# Use a reproducible sample of 100 customers.
background_data = X.sample(
    n=100,
    random_state=42
)


# =========================
# 6. TRANSFORM DATA
# =========================

# Our saved Pipeline contains:
# 1. StandardScaler
# 2. LogisticRegression
#
# SHAP will explain the classifier using
# the scaled feature values.

scaler = model.named_steps["scaler"]
classifier = model.named_steps["classifier"]

background_scaled = scaler.transform(background_data)
X_scaled = scaler.transform(X)

# Keep feature names for readable plots.
background_scaled = pd.DataFrame(
    background_scaled,
    columns=X.columns
)

X_scaled = pd.DataFrame(
    X_scaled,
    columns=X.columns
)


# =========================
# 7. CREATE SHAP EXPLAINER
# =========================

print("\nCreating SHAP LinearExplainer...")

explainer = shap.LinearExplainer(
    classifier,
    background_scaled
)

print("SHAP explainer created!")


# =========================
# 8. CALCULATE SHAP VALUES
# =========================

print("\nCalculating SHAP values...")

shap_values = explainer(X_scaled)

print("SHAP values calculated!")


# =========================
# 9. SHAP SUMMARY PLOT
# =========================

print("\nGenerating SHAP summary plot...")

plt.figure()

shap.summary_plot(
    shap_values,
    X_scaled,
    show=False
)

plt.title("SHAP Summary Plot - Customer Churn")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "shap_summary_plot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =========================
# 10. SHAP BAR PLOT
# =========================

print("Generating SHAP feature importance plot...")

plt.figure()

shap.plots.bar(
    shap_values,
    show=False
)

plt.title("SHAP Feature Importance")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "shap_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =========================
# 11. EXPLAIN ONE CUSTOMER
# =========================

print("\nGenerating individual customer explanation...")

# Example customer from our earlier prediction
customer = pd.DataFrame([{
    "age": 28,
    "premium": 6000,
    "policy_tenure": 2,
    "complaints": 4,
    "payment_delay": 45
}])

# Apply the same scaler used during training
customer_scaled = pd.DataFrame(
    scaler.transform(customer),
    columns=X.columns
)

# Calculate SHAP values for this customer
customer_shap = explainer(customer_scaled)


# =========================
# 12. SHAP WATERFALL PLOT
# =========================

shap.plots.waterfall(
    customer_shap[0],
    show=False
)

plt.title("SHAP Waterfall - Individual Customer")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "shap_waterfall_customer.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =========================
# 13. PRINT CUSTOMER PREDICTION
# =========================

prediction = model.predict(customer)[0]

churn_probability = model.predict_proba(customer)[0][1]

print("\n========================================")
print("INDIVIDUAL CUSTOMER PREDICTION")
print("========================================")

print("Customer details:")
print(customer.to_string(index=False))

print("\nPredicted result:")

if prediction == 1:
    print("CHURN")
else:
    print("RENEW")

print(f"Churn probability: {churn_probability * 100:.2f}%")


# =========================
# 14. DISPLAY FEATURE IMPORTANCE
# =========================

print("\n========================================")
print("GLOBAL SHAP FEATURE IMPORTANCE")
print("========================================")

mean_importance = np.abs(
    shap_values.values
).mean(axis=0)

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Mean Absolute SHAP Value": mean_importance
})

importance_df = importance_df.sort_values(
    by="Mean Absolute SHAP Value",
    ascending=False
)

print(importance_df.to_string(index=False))


# =========================
# 15. COMPLETION MESSAGE
# =========================

print("\n========================================")
print("SHAP EXPLAINABILITY COMPLETED!")
print("========================================")

print("\nGenerated files:")

print("1. data/shap_summary_plot.png")
print("2. data/shap_feature_importance.png")
print("3. data/shap_waterfall_customer.png")
