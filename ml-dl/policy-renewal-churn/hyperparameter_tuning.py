
# ============================================================
# POLICY RENEWAL CHURN PREDICTION
# HYPERPARAMETER TUNING - LOGISTIC REGRESSION
# ============================================================

# =========================
# 1. IMPORT LIBRARIES
# =========================

import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
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
    confusion_matrix
)


# =========================
# 2. DEFINE PROJECT PATHS
# =========================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "insurance_customers_large.csv"

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TUNED_MODEL_PATH = MODEL_DIR / "logistic_regression_tuned_pipeline.joblib"


# =========================
# 3. LOAD DATASET
# =========================

print("\nLoading insurance customer dataset...")

df = pd.read_csv(DATA_PATH)

# Separate input features and target variable
X = df.drop("churn", axis=1)
y = df["churn"]

print("Dataset shape:", df.shape)
print("Input features:", list(X.columns))


# =========================
# 4. SPLIT TRAINING AND TEST DATA
# =========================

# Keep test data separate until final evaluation.
# Stratification preserves the churn/renewal proportions.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# =========================
# 5. CREATE MACHINE LEARNING PIPELINE
# =========================

# StandardScaler scales numerical features.
# LogisticRegression performs churn classification.

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        random_state=42,
        max_iter=2000
    ))
])


# =========================
# 6. DEFINE HYPERPARAMETER GRID
# =========================

# GridSearchCV will test all combinations below.
# The double underscore accesses parameters inside
# the classifier step of our Pipeline.

param_grid = {
    "classifier__C": [0.01, 0.1, 1, 10, 100],

    "classifier__solver": ["lbfgs", "liblinear"],

    "classifier__class_weight": ["balanced", None]
}


# =========================
# 7. CONFIGURE CROSS-VALIDATION
# =========================

# StratifiedKFold preserves the class distribution
# in each fold.

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# =========================
# 8. RUN GRID SEARCH
# =========================

print("\nStarting hyperparameter tuning...")
print("Please wait. GridSearchCV is testing combinations.")

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="average_precision",
    cv=cv,
    n_jobs=-1,
    verbose=1,
    refit=True
)

# Train and evaluate different combinations using
# only the training data and cross-validation.

grid_search.fit(X_train, y_train)


# =========================
# 9. DISPLAY BEST PARAMETERS
# =========================

print("\n========================================")
print("BEST HYPERPARAMETERS")
print("========================================")

print(grid_search.best_params_)

print("\nBest cross-validation average precision:")
print(round(grid_search.best_score_, 4))


# =========================
# 10. GET BEST MODEL
# =========================

# GridSearchCV automatically refits the best model
# on the complete training dataset.

best_model = grid_search.best_estimator_


# =========================
# 11. EVALUATE ON TEST DATA
# =========================

# The test dataset has not been used during tuning.

y_pred = best_model.predict(X_test)

# Get churn probabilities for ROC-AUC and average precision.
y_probability = best_model.predict_proba(X_test)[:, 1]


accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test, y_pred, zero_division=0
)

recall = recall_score(
    y_test, y_pred, zero_division=0
)

f1 = f1_score(
    y_test, y_pred, zero_division=0
)

roc_auc = roc_auc_score(y_test, y_probability)


# =========================
# 12. PRINT EVALUATION RESULTS
# =========================

print("\n========================================")
print("TUNED MODEL TEST RESULTS")
print("========================================")

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# =========================
# 13. SAVE TUNED MODEL
# =========================

joblib.dump(best_model, TUNED_MODEL_PATH)

print("\n========================================")
print("MODEL SAVED SUCCESSFULLY")
print("========================================")

print("Saved model:", TUNED_MODEL_PATH)

print("\nHyperparameter tuning completed!")
