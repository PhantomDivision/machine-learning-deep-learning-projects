
import os
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

DATA_PATH = "data/insurance_customers_large.csv"

MODEL_PATH = "models/ann_churn_model.keras"
SCALER_PATH = "models/ann_scaler.joblib"
FEATURES_PATH = "models/feature_names.joblib"

PLOT_PATH = "data/plots/shap_customer_waterfall.png"

BACKGROUND_SIZE = 50
SHAP_SAMPLES = 100

os.makedirs("data/plots", exist_ok=True)


# --------------------------------------------------
# 2. Load dataset and saved model artifacts
# --------------------------------------------------

print("\nLoading dataset and ANN model...")

df = pd.read_csv(DATA_PATH)

feature_names = joblib.load(FEATURES_PATH)
scaler = joblib.load(SCALER_PATH)
model = load_model(MODEL_PATH)

X = df[feature_names]
y = df["churn"]

print("Dataset loaded:", X.shape)
print("Features:", feature_names)


# --------------------------------------------------
# 3. Recreate the original training split
# --------------------------------------------------

# This reproduces the same training data split used
# when the ANN was originally trained.

X_train_pool, X_test, y_train_pool, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_pool,
    y_train_pool,
    test_size=0.20,
    random_state=42,
    stratify=y_train_pool
)

print("\nTraining samples:", len(X_train))


# --------------------------------------------------
# 4. Select background customers
# --------------------------------------------------

# SHAP uses these customers as a reference to
# understand the model's normal predictions.

background = X_train.sample(
    n=min(BACKGROUND_SIZE, len(X_train)),
    random_state=42
).to_numpy(dtype=float)


# --------------------------------------------------
# 5. Define ANN prediction function
# --------------------------------------------------

# SHAP supplies raw customer feature values.
# We apply the same scaler used during ANN training
# before passing the data into the model.

def predict_probability(raw_data):

    raw_df = pd.DataFrame(
        raw_data,
        columns=feature_names
    )

    scaled_data = scaler.transform(raw_df)

    predictions = model.predict(
        scaled_data,
        verbose=0
    )

    return predictions.reshape(-1)


# --------------------------------------------------
# 6. Define the customer to explain
# --------------------------------------------------

customer = pd.DataFrame(
    [{
        "age": 28,
        "premium": 6000,
        "policy_tenure": 2,
        "complaints": 4,
        "payment_delay": 45
    }]
)

# Ensure the feature order matches model training.
customer = customer[feature_names]

customer_values = customer.to_numpy(dtype=float)


# --------------------------------------------------
# 7. Get the ANN prediction
# --------------------------------------------------

churn_probability = float(
    predict_probability(customer_values)[0]
)

threshold = joblib.load("models/ann_threshold.joblib")

print("\n" + "=" * 50)
print("CUSTOMER DETAILS")
print("=" * 50)

print(customer.to_string(index=False))

print(f"\nChurn probability: {churn_probability:.2%}")
print(f"Classification threshold: {threshold:.2f}")

if churn_probability >= threshold:
    print("Prediction: LIKELY TO CHURN")
else:
    print("Prediction: LIKELY TO RENEW")


# --------------------------------------------------
# 8. Initialize SHAP KernelExplainer
# --------------------------------------------------

print("\nInitializing SHAP KernelExplainer...")

explainer = shap.KernelExplainer(
    predict_probability,
    background
)


# --------------------------------------------------
# 9. Calculate SHAP values
# --------------------------------------------------

print("\nCalculating SHAP values...")
print("This may take a little time.")

shap_values = explainer.shap_values(
    customer_values,
    nsamples=SHAP_SAMPLES
)

# Handle different SHAP output formats.
if isinstance(shap_values, list):
    shap_values = shap_values[0]

shap_values = np.asarray(shap_values)

if shap_values.ndim == 2:
    shap_values = shap_values[0]

shap_values = shap_values.reshape(-1)

base_value = float(
    np.asarray(explainer.expected_value).reshape(-1)[0]
)


# --------------------------------------------------
# 10. Display feature contributions
# --------------------------------------------------

print("\n" + "=" * 50)
print("SHAP FEATURE CONTRIBUTIONS")
print("=" * 50)

contributions = pd.DataFrame({
    "Feature": feature_names,
    "Customer Value": customer_values[0],
    "SHAP Contribution": shap_values
})

contributions["Absolute Contribution"] = (
    contributions["SHAP Contribution"].abs()
)

contributions = contributions.sort_values(
    by="Absolute Contribution",
    ascending=False
)

print(contributions.to_string(index=False))

print(f"\nSHAP baseline probability: {base_value:.4f}")
print(f"Sum of SHAP contributions: {shap_values.sum():.4f}")
print(
    "Reconstructed probability: "
    f"{base_value + shap_values.sum():.4f}"
)
print(f"ANN probability: {churn_probability:.4f}")


# --------------------------------------------------
# 11. Generate SHAP waterfall plot
# --------------------------------------------------

explanation = shap.Explanation(
    values=shap_values,
    base_values=base_value,
    data=customer_values[0],
    feature_names=feature_names
)

plt.figure(figsize=(10, 6))

shap.plots.waterfall(
    explanation,
    show=False
)

plt.title(
    "SHAP Explanation - Insurance Customer Churn",
    fontsize=13
)

plt.tight_layout()

plt.savefig(
    PLOT_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nSHAP waterfall plot saved to:", PLOT_PATH)

print("\nSHAP local explanation completed successfully.")
