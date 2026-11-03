
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

SUMMARY_PLOT_PATH = "data/plots/shap_global_summary.png"
BAR_PLOT_PATH = "data/plots/shap_global_importance.png"

BACKGROUND_SIZE = 50
EXPLANATION_SIZE = 100
SHAP_SAMPLES = 100

os.makedirs("data/plots", exist_ok=True)


# --------------------------------------------------
# 2. Load dataset and trained artifacts
# --------------------------------------------------

print("\nLoading dataset and ANN model...")

df = pd.read_csv(DATA_PATH)

feature_names = joblib.load(FEATURES_PATH)
scaler = joblib.load(SCALER_PATH)
model = load_model(MODEL_PATH)

X = df[feature_names]
y = df["churn"]

print("Dataset shape:", X.shape)
print("Features:", feature_names)


# --------------------------------------------------
# 3. Recreate original train/test split
# --------------------------------------------------

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

print("\nTraining customers:", len(X_train))
print("Test customers:", len(X_test))


# --------------------------------------------------
# 4. Select background and explanation samples
# --------------------------------------------------

background_df = X_train.sample(
    n=min(BACKGROUND_SIZE, len(X_train)),
    random_state=42
)

explanation_df = X_test.sample(
    n=min(EXPLANATION_SIZE, len(X_test)),
    random_state=42
)

background = background_df.to_numpy(dtype=float)
explanation_data = explanation_df.to_numpy(dtype=float)

print("\nBackground sample size:", len(background))
print("Customers to explain:", len(explanation_data))


# --------------------------------------------------
# 5. Define ANN prediction function
# --------------------------------------------------

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
# 6. Initialize SHAP KernelExplainer
# --------------------------------------------------

print("\nInitializing SHAP KernelExplainer...")

explainer = shap.KernelExplainer(
    predict_probability,
    background
)


# --------------------------------------------------
# 7. Calculate global SHAP values
# --------------------------------------------------

print("\nCalculating SHAP values for test customers...")
print("This may take some time.")

shap_values = explainer.shap_values(
    explanation_data,
    nsamples=SHAP_SAMPLES
)

# Handle different SHAP output formats.
if isinstance(shap_values, list):
    shap_values = shap_values[0]

shap_values = np.asarray(shap_values)

# Expected shape: (number of customers, number of features)
if shap_values.ndim == 3:
    shap_values = shap_values[:, :, 0]

print("\nSHAP values shape:", shap_values.shape)


# --------------------------------------------------
# 8. Calculate global feature importance
# --------------------------------------------------

mean_abs_shap = np.mean(
    np.abs(shap_values),
    axis=0
)

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Mean Absolute SHAP": mean_abs_shap
})

importance_df = importance_df.sort_values(
    by="Mean Absolute SHAP",
    ascending=False
)

print("\n" + "=" * 55)
print("GLOBAL SHAP FEATURE IMPORTANCE")
print("=" * 55)

print(importance_df.to_string(index=False))


# --------------------------------------------------
# 9. Generate SHAP Summary (Beeswarm) Plot
# --------------------------------------------------

print("\nGenerating SHAP summary plot...")

global_explanation = shap.Explanation(
    values=shap_values,
    data=explanation_data,
    feature_names=feature_names
)

plt.figure(figsize=(10, 6))

shap.plots.beeswarm(
    global_explanation,
    max_display=len(feature_names),
    show=False
)

plt.title(
    "Global SHAP Summary - ANN Churn Model",
    fontsize=13
)

plt.tight_layout()

plt.savefig(
    SUMMARY_PLOT_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# 10. Generate SHAP Feature Importance Bar Plot
# --------------------------------------------------

print("\nGenerating SHAP importance bar plot...")

plt.figure(figsize=(10, 6))

shap.plots.bar(
    global_explanation,
    max_display=len(feature_names),
    show=False
)

plt.title(
    "Global SHAP Feature Importance - ANN",
    fontsize=13
)

plt.tight_layout()

plt.savefig(
    BAR_PLOT_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# 11. Save global SHAP values
# --------------------------------------------------

shap_df = pd.DataFrame(
    shap_values,
    columns=feature_names
)

shap_df.to_csv(
    "data/global_shap_values.csv",
    index=False
)

print("\n" + "=" * 55)
print("GLOBAL SHAP EXPLANATION COMPLETED")
print("=" * 55)

print("\nGenerated files:")
print("1.", SUMMARY_PLOT_PATH)
print("2.", BAR_PLOT_PATH)
print("3. data/global_shap_values.csv")
