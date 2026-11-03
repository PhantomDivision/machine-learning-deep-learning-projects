# ann_model.py

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os
import random
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping


# ============================================================
# 2. CONFIGURATION
# ============================================================

RANDOM_STATE = 42

DATA_PATH = "data/insurance_customers_large.csv"
MODEL_DIR = "models"
PLOT_DIR = "data/plots"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

FEATURE_COLUMNS = [
    "age",
    "premium",
    "policy_tenure",
    "complaints",
    "payment_delay"
]

TARGET_COLUMN = "churn"

EPOCHS = 100
BATCH_SIZE = 32

DEFAULT_THRESHOLD = 0.50


# ============================================================
# 3. REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# ============================================================
# 4. LOAD DATASET
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)

required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

if df[required_columns].isnull().any().any():
    raise ValueError(
        "Dataset contains missing values. "
        "Please handle them before training."
    )

X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMN]

print("\nChurn distribution:")
print(y.value_counts())

print("\nChurn percentage:")
print(y.value_counts(normalize=True) * 100)


# ============================================================
# 5. TRAIN, VALIDATION AND TEST SPLIT
# ============================================================

print("\nSplitting dataset...")

# First split: 80% training pool and 20% test data

X_train_pool, X_test, y_train_pool, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

# Second split: 80% training and 20% validation
# Final proportions: 64% train, 16% validation, 20% test

X_train, X_val, y_train, y_val = train_test_split(
    X_train_pool,
    y_train_pool,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y_train_pool
)

print("\nDataset split:")
print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Test samples:", len(X_test))

print("\nChurn distribution in each dataset:")
print("Training:\n", y_train.value_counts())
print("Validation:\n", y_val.value_counts())
print("Test:\n", y_test.value_counts())


# ============================================================
# 6. FEATURE SCALING
# ============================================================

print("\nScaling features...")

scaler = StandardScaler()

# Fit scaler ONLY on training data

X_train_scaled = scaler.fit_transform(X_train)

# Apply the same scaler to validation and test data

X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# 7. HANDLE CLASS IMBALANCE
# ============================================================

print("\nCalculating class weights...")

classes = np.unique(y_train)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weight_dict = {
    int(cls): float(weight)
    for cls, weight in zip(classes, class_weights)
}

print("Class weights:", class_weight_dict)


# ============================================================
# 8. BUILD ANN MODEL
# ============================================================

print("\nBuilding ANN model...")

model = Sequential([
    Input(shape=(len(FEATURE_COLUMNS),)),

    Dense(64, activation="relu"),

    Dropout(0.2),

    Dense(32, activation="relu"),

    Dropout(0.2),

    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="binary_crossentropy",

    metrics=[
        "accuracy",

        tf.keras.metrics.Precision(name="precision"),

        tf.keras.metrics.Recall(name="recall"),

        tf.keras.metrics.AUC(name="roc_auc"),

        tf.keras.metrics.AUC(
            name="pr_auc",
            curve="PR"
        )
    ]
)

model.summary()


# ============================================================
# 9. EARLY STOPPING
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=12,
    restore_best_weights=True,
    verbose=1
)


# ============================================================
# 10. TRAIN ANN MODEL
# ============================================================

print("\nStarting ANN training...")

history = model.fit(
    X_train_scaled,
    y_train,

    validation_data=(
        X_val_scaled,
        y_val
    ),

    epochs=EPOCHS,
    batch_size=BATCH_SIZE,

    class_weight=class_weight_dict,

    callbacks=[early_stopping],

    verbose=1
)


# ============================================================
# 11. PLOT TRAINING ACCURACY
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("ANN Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    f"{PLOT_DIR}/ann_accuracy.png"
)

plt.close()


# ============================================================
# 12. PLOT TRAINING LOSS
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("ANN Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    f"{PLOT_DIR}/ann_loss.png"
)

plt.close()


# ============================================================
# 13. GET VALIDATION PROBABILITIES
# ============================================================

print("\nGenerating validation predictions...")

# These are probabilities between 0 and 1

val_probabilities = model.predict(
    X_val_scaled,
    verbose=0
).ravel()

y_val_array = np.asarray(y_val)


# ============================================================
# 14. THRESHOLD TUNING USING VALIDATION DATA
# ============================================================

print("\nStarting threshold tuning...")

thresholds = np.arange(
    0.05,
    0.96,
    0.01
)

threshold_results = []

for threshold in thresholds:

    val_predictions = (
        val_probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_val_array,
        val_predictions,
        zero_division=0
    )

    recall = recall_score(
        y_val_array,
        val_predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_val_array,
        val_predictions,
        zero_division=0
    )

    threshold_results.append({
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })


threshold_df = pd.DataFrame(threshold_results)

# Select threshold with maximum validation F1-score.
# If multiple thresholds have the same F1, prefer higher recall.

best_result = max(
    threshold_results,
    key=lambda result: (
        result["f1_score"],
        result["recall"]
    )
)

best_threshold = best_result["threshold"]

print("\nThreshold tuning completed!")

print("\nBest validation threshold:")
print(f"Threshold : {best_threshold:.2f}")
print(f"Precision : {best_result['precision']:.4f}")
print(f"Recall    : {best_result['recall']:.4f}")
print(f"F1-score  : {best_result['f1_score']:.4f}")


# ============================================================
# 15. PLOT THRESHOLD TUNING RESULTS
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(
    threshold_df["threshold"],
    threshold_df["precision"],
    label="Precision"
)

plt.plot(
    threshold_df["threshold"],
    threshold_df["recall"],
    label="Recall"
)

plt.plot(
    threshold_df["threshold"],
    threshold_df["f1_score"],
    label="F1-score",
    linewidth=2
)

plt.axvline(
    best_threshold,
    linestyle="--",
    label=f"Best Threshold: {best_threshold:.2f}"
)

plt.title("Validation Metrics at Different Thresholds")
plt.xlabel("Classification Threshold")
plt.ylabel("Score")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    f"{PLOT_DIR}/ann_threshold_tuning.png"
)

plt.close()


# ============================================================
# 16. GENERATE TEST PROBABILITIES
# ============================================================

print("\nGenerating test predictions...")

# The test data is used only now, after threshold selection.

test_probabilities = model.predict(
    X_test_scaled,
    verbose=0
).ravel()

y_test_array = np.asarray(y_test)


# ============================================================
# 17. EVALUATION FUNCTION
# ============================================================

def evaluate_model(y_true, probabilities, threshold):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    metrics = {
        "accuracy": accuracy_score(
            y_true,
            predictions
        ),

        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "f1_score": f1_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "roc_auc": roc_auc_score(
            y_true,
            probabilities
        ),

        "pr_auc": average_precision_score(
            y_true,
            probabilities
        )
    }

    cm = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    )

    return metrics, cm, predictions


# ============================================================
# 18. COMPARE DEFAULT AND TUNED THRESHOLDS
# ============================================================

print("\nEvaluating default threshold...")

default_metrics, default_cm, default_predictions = (
    evaluate_model(
        y_test_array,
        test_probabilities,
        DEFAULT_THRESHOLD
    )
)

print("\nEvaluating tuned threshold...")

tuned_metrics, tuned_cm, tuned_predictions = (
    evaluate_model(
        y_test_array,
        test_probabilities,
        best_threshold
    )
)


# ============================================================
# 19. PRINT METRIC COMPARISON
# ============================================================

print("\n========================================")
print("ANN TEST METRIC COMPARISON")
print("========================================")

comparison_df = pd.DataFrame(
    {
        "Default Threshold (0.50)": default_metrics,
        "Tuned Threshold": tuned_metrics
    }
)

print(comparison_df.round(4))


print("\n========================================")
print("DEFAULT THRESHOLD CONFUSION MATRIX")
print("========================================")

print(default_cm)


print("\n========================================")
print("TUNED THRESHOLD CONFUSION MATRIX")
print("========================================")

print(tuned_cm)


# ============================================================
# 20. CLASSIFICATION REPORT
# ============================================================

print("\n========================================")
print("TUNED THRESHOLD CLASSIFICATION REPORT")
print("========================================")

print(
    classification_report(
        y_test_array,
        tuned_predictions,
        target_names=[
            "Renewed (0)",
            "Churned (1)"
        ],
        zero_division=0
    )
)


# ============================================================
# 21. PLOT TUNED CONFUSION MATRIX
# ============================================================

plt.figure(figsize=(7, 5))

sns.heatmap(
    tuned_cm,
    annot=True,
    fmt="d",
    cmap="Blues",

    xticklabels=[
        "Predicted Renewed",
        "Predicted Churn"
    ],

    yticklabels=[
        "Actual Renewed",
        "Actual Churn"
    ]
)

plt.title(
    f"ANN Confusion Matrix (Threshold = {best_threshold:.2f})"
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.tight_layout()

plt.savefig(
    f"{PLOT_DIR}/ann_confusion_matrix.png"
)

plt.close()


# ============================================================
# 22. ROC CURVE
# ============================================================

fpr, tpr, roc_thresholds = roc_curve(
    y_test_array,
    test_probabilities
)

plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label=f"ROC-AUC = {tuned_metrics['roc_auc']:.4f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.title("ANN ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    f"{PLOT_DIR}/ann_roc_curve.png"
)

plt.close()


# ============================================================
# 23. PRECISION-RECALL CURVE
# ============================================================

precision_values, recall_values, pr_thresholds = (
    precision_recall_curve(
        y_test_array,
        test_probabilities
    )
)

plt.figure(figsize=(7, 5))

plt.plot(
    recall_values,
    precision_values,
    label=f"PR-AUC = {tuned_metrics['pr_auc']:.4f}"
)

plt.title("ANN Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    f"{PLOT_DIR}/ann_precision_recall_curve.png"
)

plt.close()


# ============================================================
# 24. SAVE MODEL AND PREPROCESSING ARTIFACTS
# ============================================================

print("\nSaving model and preprocessing artifacts...")

# Save trained ANN

model.save(
    f"{MODEL_DIR}/ann_churn_model.keras"
)

# Save scaler

joblib.dump(
    scaler,
    f"{MODEL_DIR}/ann_scaler.joblib"
)

# Save feature names

joblib.dump(
    FEATURE_COLUMNS,
    f"{MODEL_DIR}/feature_names.joblib"
)

# Save tuned classification threshold

joblib.dump(
    best_threshold,
    f"{MODEL_DIR}/ann_threshold.joblib"
)


# ============================================================
# 25. FINAL SUMMARY
# ============================================================

print("\n========================================")
print("ANN TRAINING AND EVALUATION COMPLETED")
print("========================================")

print("\nSelected threshold:", round(best_threshold, 2))

print("\nFinal test metrics using tuned threshold:")

for metric, value in tuned_metrics.items():

    print(f"{metric}: {value:.4f}")


print("\nSaved model:")
print(f"{MODEL_DIR}/ann_churn_model.keras")

print("\nSaved scaler:")
print(f"{MODEL_DIR}/ann_scaler.joblib")

print("\nSaved feature names:")
print(f"{MODEL_DIR}/feature_names.joblib")

print("\nSaved threshold:")
print(f"{MODEL_DIR}/ann_threshold.joblib")

print("\nPlots saved in:")
print(PLOT_DIR)

print("\n========================================")