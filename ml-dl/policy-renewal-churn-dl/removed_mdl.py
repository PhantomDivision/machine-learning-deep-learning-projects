
"""
=========================================================
PROJECT: POLICY RENEWAL CHURN PREDICTION
MODEL: ARTIFICIAL NEURAL NETWORK (ANN)
=========================================================

This script performs:

1. Dataset loading
2. Feature and target separation
3. Train / validation / test splitting
4. Feature scaling
5. Class weight calculation
6. ANN architecture creation
7. ANN model training
8. Early stopping
9. Model evaluation
10. Confusion matrix
11. ROC curve and Precision-Recall curve
12. Training history visualization
13. Model and scaler persistence
"""

# =========================================================
# 1. IMPORT LIBRARIES
# =========================================================

import os
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import joblib

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
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# =========================================================
# 2. CONFIGURATION
# =========================================================

DATA_PATH = "data/insurance_customers_large.csv"

MODEL_DIR = "models"
PLOT_DIR = "data/plots"

MODEL_PATH = f"{MODEL_DIR}/ann_churn_model.keras"
SCALER_PATH = f"{MODEL_DIR}/ann_scaler.joblib"
FEATURE_PATH = f"{MODEL_DIR}/feature_names.joblib"

FEATURE_COLUMNS = [
    "age",
    "premium",
    "policy_tenure",
    "complaints",
    "payment_delay"
]

TARGET_COLUMN = "churn"

RANDOM_STATE = 42

EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# Prediction threshold.
# A customer is classified as churned when probability >= 0.5.
CLASSIFICATION_THRESHOLD = 0.5

# Reproducibility
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")


# =========================================================
# 3. LOAD DATASET
# =========================================================

def load_dataset():

    print("\n========== LOADING DATASET ==========")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if df[required_columns].isnull().any().any():
        raise ValueError(
            "Dataset contains missing values."
        )

    if not set(df[TARGET_COLUMN].unique()).issubset({0, 1}):
        raise ValueError(
            "Target column must contain only 0 and 1."
        )

    print("Dataset loaded successfully!")
    print(f"Dataset shape: {df.shape}")

    return df


# =========================================================
# 4. SEPARATE FEATURES AND TARGET
# =========================================================

def separate_features_target(df):

    print("\n========== SEPARATING FEATURES AND TARGET ==========")

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    print("\nTarget distribution:")
    print(y.value_counts().sort_index())

    return X, y


# =========================================================
# 5. TRAIN / VALIDATION / TEST SPLIT
# =========================================================

def split_data(X, y):

    print("\n========== SPLITTING DATA ==========")

    # First split:
    # 80% training pool and 20% testing data.

    X_train_full, X_test, y_train_full, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=RANDOM_STATE,
            stratify=y
        )
    )

    # Second split:
    # 80% of the training pool for training,
    # 20% for validation.
    #
    # Final proportions:
    # 64% training
    # 16% validation
    # 20% testing

    X_train, X_val, y_train, y_val = (
        train_test_split(
            X_train_full,
            y_train_full,
            test_size=0.20,
            random_state=RANDOM_STATE,
            stratify=y_train_full
        )
    )

    print(f"Training records:   {len(X_train)}")
    print(f"Validation records: {len(X_val)}")
    print(f"Testing records:    {len(X_test)}")

    print("\nTraining target distribution:")
    print(y_train.value_counts().sort_index())

    print("\nValidation target distribution:")
    print(y_val.value_counts().sort_index())

    print("\nTesting target distribution:")
    print(y_test.value_counts().sort_index())

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )


# =========================================================
# 6. FEATURE SCALING
# =========================================================

def scale_features(X_train, X_val, X_test):

    print("\n========== FEATURE SCALING ==========")

    scaler = StandardScaler()

    # Fit only on training data.
    X_train_scaled = scaler.fit_transform(X_train)

    # Transform validation and testing data.
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print("StandardScaler applied successfully!")

    print("\nTraining feature means:")
    print(np.round(X_train_scaled.mean(axis=0), 3))

    print("\nTraining feature standard deviations:")
    print(np.round(X_train_scaled.std(axis=0), 3))

    return (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        scaler
    )


# =========================================================
# 7. CLASS WEIGHT CALCULATION
# =========================================================

def calculate_class_weights(y_train):

    print("\n========== CALCULATING CLASS WEIGHTS ==========")

    classes = np.unique(y_train)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )

    class_weights = {
        int(class_label): float(weight)
        for class_label, weight in zip(classes, weights)
    }

    print("Class weights:")
    print(class_weights)

    return class_weights


# =========================================================
# 8. BUILD ANN ARCHITECTURE
# =========================================================

def build_ann(input_features):

    print("\n========== BUILDING ANN MODEL ==========")

    model = Sequential(
        [
            Input(shape=(input_features,)),

            Dense(
                64,
                activation="relu",
                name="hidden_layer_1"
            ),

            Dropout(
                0.2,
                name="dropout_layer_1"
            ),

            Dense(
                32,
                activation="relu",
                name="hidden_layer_2"
            ),

            Dropout(
                0.2,
                name="dropout_layer_2"
            ),

            Dense(
                1,
                activation="sigmoid",
                name="output_layer"
            )
        ]
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    )

    model.compile(
        optimizer=optimizer,
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

    print("\nANN architecture:")
    model.summary()

    return model


# =========================================================
# 9. TRAIN ANN MODEL
# =========================================================

def train_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    class_weights
):

    print("\n========== TRAINING ANN MODEL ==========")

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=12,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=[early_stopping],
        verbose=1
    )

    print("\nTraining completed!")

    return history


# =========================================================
# 10. PLOT TRAINING HISTORY
# =========================================================

def plot_training_history(history):

    print("\n========== PLOTTING TRAINING HISTORY ==========")

    # Accuracy plot

    plt.figure(figsize=(8, 5))

    plt.plot(
        history.history["accuracy"],
        label="Training Accuracy"
    )

    plt.plot(
        history.history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.title("ANN Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        f"{PLOT_DIR}/ann_accuracy.png",
        dpi=300
    )

    plt.close()

    # Loss plot

    plt.figure(figsize=(8, 5))

    plt.plot(
        history.history["loss"],
        label="Training Loss"
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss"
    )

    plt.title("ANN Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        f"{PLOT_DIR}/ann_loss.png",
        dpi=300
    )

    plt.close()

    print("Training plots saved successfully!")


# =========================================================
# 11. EVALUATE ANN MODEL
# =========================================================

def evaluate_model(model, X_test, y_test):

    print("\n========== MODEL EVALUATION ==========")

    # Predict churn probabilities.
    y_probability = model.predict(
        X_test,
        verbose=0
    ).ravel()

    # Convert probabilities into binary predictions.
    y_prediction = (
        y_probability >= CLASSIFICATION_THRESHOLD
    ).astype(int)

    # Calculate classification metrics.
    accuracy = accuracy_score(
        y_test,
        y_prediction
    )

    precision = precision_score(
        y_test,
        y_prediction,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_prediction,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_prediction,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )

    pr_auc = average_precision_score(
        y_test,
        y_probability
    )

    print("\n========== CLASSIFICATION METRICS ==========")

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")

    print("\n========== CLASSIFICATION REPORT ==========")

    print(
        classification_report(
            y_test,
            y_prediction,
            target_names=[
                "Renewed",
                "Churned"
            ],
            zero_division=0
        )
    )

    # Return predictions and metrics for further use.
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc
    }

    return y_prediction, y_probability, metrics


# =========================================================
# 12. CONFUSION MATRIX
# =========================================================

def plot_confusion_matrix(y_test, y_prediction):

    print("\n========== CONFUSION MATRIX ==========")

    cm = confusion_matrix(
        y_test,
        y_prediction
    )

    print(cm)

    plt.figure(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Renewed", "Churned"],
        yticklabels=["Renewed", "Churned"]
    )

    plt.title("ANN Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()

    plt.savefig(
        f"{PLOT_DIR}/ann_confusion_matrix.png",
        dpi=300
    )

    plt.close()


# =========================================================
# 13. ROC CURVE
# =========================================================

def plot_roc_curve(y_test, y_probability):

    print("\n========== ROC CURVE ==========")

    fpr, tpr, thresholds = roc_curve(
        y_test,
        y_probability
    )

    auc_value = roc_auc_score(
        y_test,
        y_probability
    )

    plt.figure(figsize=(7, 5))

    plt.plot(
        fpr,
        tpr,
        label=f"ANN ROC-AUC = {auc_value:.4f}"
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
    plt.tight_layout()

    plt.savefig(
        f"{PLOT_DIR}/ann_roc_curve.png",
        dpi=300
    )

    plt.close()


# =========================================================
# 14. PRECISION-RECALL CURVE
# =========================================================

def plot_precision_recall_curve(y_test, y_probability):

    print("\n========== PRECISION-RECALL CURVE ==========")

    precision_values, recall_values, thresholds = (
        precision_recall_curve(
            y_test,
            y_probability
        )
    )

    ap_value = average_precision_score(
        y_test,
        y_probability
    )

    plt.figure(figsize=(7, 5))

    plt.plot(
        recall_values,
        precision_values,
        label=f"ANN PR-AUC = {ap_value:.4f}"
    )

    plt.title("ANN Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        f"{PLOT_DIR}/ann_precision_recall_curve.png",
        dpi=300
    )

    plt.close()


# =========================================================
# 15. SAVE MODEL AND PREPROCESSING OBJECTS
# =========================================================

def save_artifacts(model, scaler):

    print("\n========== SAVING MODEL ==========")

    model.save(MODEL_PATH)

    joblib.dump(
        scaler,
        SCALER_PATH
    )

    joblib.dump(
        FEATURE_COLUMNS,
        FEATURE_PATH
    )

    print(f"ANN model saved: {MODEL_PATH}")
    print(f"Scaler saved: {SCALER_PATH}")
    print(f"Feature names saved: {FEATURE_PATH}")


# =========================================================
# 16. MAIN EXECUTION
# =========================================================

def main():

    print("\n==============================================")
    print(" POLICY RENEWAL CHURN - ANN TRAINING")
    print("==============================================")

    # Step 1: Load and prepare dataset.
    df = load_dataset()

    X, y = separate_features_target(df)

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    ) = split_data(X, y)

    (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        scaler
    ) = scale_features(
        X_train,
        X_val,
        X_test
    )

    class_weights = calculate_class_weights(y_train)

    # Step 2: Build ANN.
    model = build_ann(
        input_features=X_train_scaled.shape[1]
    )

    # Step 3: Train ANN.
    history = train_model(
        model,
        X_train_scaled,
        y_train.to_numpy(),
        X_val_scaled,
        y_val.to_numpy(),
        class_weights
    )

    # Step 4: Plot training history.
    plot_training_history(history)

    # Step 5: Evaluate on unseen test data.
    (
        y_prediction,
        y_probability,
        metrics
    ) = evaluate_model(
        model,
        X_test_scaled,
        y_test.to_numpy()
    )

    # Step 6: Evaluation plots.
    plot_confusion_matrix(
        y_test.to_numpy(),
        y_prediction
    )

    plot_roc_curve(
        y_test.to_numpy(),
        y_probability
    )

    plot_precision_recall_curve(
        y_test.to_numpy(),
        y_probability
    )

    # Step 7: Save model and scaler.
    save_artifacts(
        model,
        scaler
    )

    print("\n==============================================")
    print(" ANN TRAINING AND EVALUATION COMPLETED")
    print("==============================================")

    print("\nFinal test metrics:")

    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    print(f"\nModel location: {MODEL_PATH}")
    print(f"Plots location: {PLOT_DIR}/")


# =========================================================
# 17. SCRIPT ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()
