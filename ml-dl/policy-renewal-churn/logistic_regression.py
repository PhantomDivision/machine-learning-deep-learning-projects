
# ============================================================
# PROJECT: POLICY RENEWAL CHURN PREDICTION
# MODEL: LOGISTIC REGRESSION
# PURPOSE: Predict insurance customer churn
# ============================================================


# ============================================================
# 1. IMPORT REQUIRED LIBRARIES
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)


# ============================================================
# 2. DEFINE PROJECT DIRECTORIES
# ============================================================

# Get the directory containing this Python file
BASE_DIR = Path(__file__).resolve().parent

# Dataset location
DATA_PATH = BASE_DIR / "data" / "insurance_customers_large.csv"

# Directory to save evaluation charts
OUTPUT_DIR = BASE_DIR / "data"

# Directory to save trained models
MODEL_DIR = BASE_DIR / "models"

# Create directories if they do not exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Saved model location
MODEL_PATH = MODEL_DIR / "logistic_regression_pipeline.joblib"


# ============================================================
# 3. LOAD DATASET
# ============================================================

print("\n" + "=" * 60)
print("INSURANCE POLICY RENEWAL CHURN PREDICTION")
print("MODEL: LOGISTIC REGRESSION")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully!")

print("Dataset shape:", df.shape)

print("\nFirst 5 records:")
print(df.head())


# ============================================================
# 4. SEPARATE FEATURES AND TARGET
# ============================================================

# Input features used for prediction
X = df.drop("churn", axis=1)

# Target variable
# 0 = Customer renews
# 1 = Customer churns
y = df["churn"]

print("\nInput features:")
print(X.columns.tolist())

print("\nTarget distribution:")
print(y.value_counts())

print("\nChurn percentage:")
print((y.value_counts(normalize=True) * 100).round(2))


# ============================================================
# 5. SPLIT DATA INTO TRAINING AND TESTING
# ============================================================

# 80% training data
# 20% testing data
#
# random_state ensures reproducible results.
# stratify preserves the churn distribution in both sets.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTrain-test split completed!")

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())


# ============================================================
# 6. CREATE LOGISTIC REGRESSION PIPELINE
# ============================================================

# StandardScaler:
# Scales numerical features to comparable ranges.
#
# LogisticRegression:
# Our classification algorithm.
#
# class_weight="balanced":
# Gives greater importance to the minority churn class.
#
# Pipeline:
# Combines preprocessing and model training.
# Prevents data leakage because the scaler is fitted
# using training data only.

model = Pipeline([
    ("scaler", StandardScaler()),

    ("classifier", LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    ))
])


# ============================================================
# 7. TRAIN MODEL
# ============================================================

print("\nTraining Logistic Regression model...")

model.fit(X_train, y_train)

print("Model training completed successfully!")


# ============================================================
# 8. SAVE TRAINED MODEL
# ============================================================

# Save the entire pipeline, including StandardScaler
# and Logistic Regression classifier.
#
# Our future API will load this file directly.
# It will not need to train the model again.

joblib.dump(model, MODEL_PATH)

print("\nTrained model saved successfully!")
print("Model location:", MODEL_PATH)


# ============================================================
# 9. MAKE TEST DATA PREDICTIONS
# ============================================================

# Predict class labels
# 0 = Renewed
# 1 = Churned

y_pred = model.predict(X_test)

# Predict probability of churn
# [:, 1] selects the probability for class 1 (churn)

y_probability = model.predict_proba(X_test)[:, 1]

print("\nPredictions completed successfully!")


# ============================================================
# 10. CALCULATE PERFORMANCE METRICS
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


# ============================================================
# 11. DISPLAY MODEL PERFORMANCE
# ============================================================

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION MODEL PERFORMANCE")
print("=" * 60)

print(f"\nAccuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Renewed", "Churned"],
        zero_division=0
    )
)


# ============================================================
# 12. GENERATE CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

cm_display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Renewed", "Churned"]
)

cm_display.plot(
    cmap="Blues",
    values_format="d"
)

plt.title("Logistic Regression - Confusion Matrix")
plt.tight_layout()

# Save confusion matrix
CONFUSION_MATRIX_PATH = (
    OUTPUT_DIR / "logistic_regression_confusion_matrix.png"
)

plt.savefig(
    CONFUSION_MATRIX_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nConfusion matrix saved:", CONFUSION_MATRIX_PATH)


# ============================================================
# 13. GENERATE ROC CURVE
# ============================================================

RocCurveDisplay.from_predictions(
    y_test,
    y_probability,
    name="Logistic Regression"
)

plt.title("Logistic Regression - ROC Curve")
plt.tight_layout()

# Save ROC curve
ROC_CURVE_PATH = OUTPUT_DIR / "logistic_regression_roc_curve.png"

plt.savefig(
    ROC_CURVE_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nROC curve saved:", ROC_CURVE_PATH)


# ============================================================
# 14. FINAL EXECUTION SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION EXECUTION COMPLETED!")
print("=" * 60)

print("\nGenerated files:")

print("- models/logistic_regression_pipeline.joblib")

print("- data/logistic_regression_confusion_matrix.png")

print("- data/logistic_regression_roc_curve.png")

print("\nModel is ready for new customer predictions!")

print("\nNext stage: Build a prediction script and REST API.")

print("=" * 60)
