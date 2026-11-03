
# predict_customer.py

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf


# ============================================================
# 2. MODEL FILE PATHS
# ============================================================

MODEL_PATH = "models/ann_churn_model.keras"
SCALER_PATH = "models/ann_scaler.joblib"
FEATURE_PATH = "models/feature_names.joblib"
THRESHOLD_PATH = "models/ann_threshold.joblib"


# ============================================================
# 3. LOAD SAVED ARTIFACTS
# ============================================================

print("\nLoading ANN model and preprocessing artifacts...")

model = tf.keras.models.load_model(MODEL_PATH)

scaler = joblib.load(SCALER_PATH)

feature_names = joblib.load(FEATURE_PATH)

threshold = joblib.load(THRESHOLD_PATH)

print("All artifacts loaded successfully!")


# ============================================================
# 4. GET CUSTOMER INPUT
# ============================================================

def get_customer_input():

    print("\n===================================")
    print("ENTER CUSTOMER DETAILS")
    print("===================================")

    customer_data = {}

    for feature in feature_names:

        while True:

            try:
                value = float(
                    input(f"Enter {feature}: ")
                )

                if not np.isfinite(value):
                    print("Please enter a valid finite number.")
                    continue

                customer_data[feature] = value
                break

            except ValueError:
                print("Invalid input. Please enter a number.")

    return customer_data


# ============================================================
# 5. PREPROCESS CUSTOMER DATA
# ============================================================

def preprocess_customer(customer_data):

    # Convert dictionary into a DataFrame
    # using the exact training feature order.

    customer_df = pd.DataFrame(
        [customer_data],
        columns=feature_names
    )

    # Apply the saved scaler.
    # Do not fit a new scaler.

    scaled_data = scaler.transform(customer_df)

    return scaled_data


# ============================================================
# 6. PREDICT CUSTOMER CHURN
# ============================================================

def predict_customer(customer_data):

    scaled_data = preprocess_customer(customer_data)

    # Generate churn probability

    churn_probability = model.predict(
        scaled_data,
        verbose=0
    )[0][0]

    # Apply the saved threshold

    if churn_probability >= threshold:

        prediction = "LIKELY TO CHURN"

    else:

        prediction = "LIKELY TO RENEW"

    return churn_probability, prediction


# ============================================================
# 7. DISPLAY PREDICTION
# ============================================================

def display_result(
    customer_data,
    churn_probability,
    prediction
):

    print("\n===================================")
    print("CUSTOMER CHURN PREDICTION")
    print("===================================")

    print("\nCustomer details:")

    for feature, value in customer_data.items():

        print(f"{feature}: {value}")

    print("\nPrediction results:")

    print(
        f"Churn probability: "
        f"{churn_probability * 100:.2f}%"
    )

    print(
        f"Classification threshold: "
        f"{threshold:.2f}"
    )

    print(f"Prediction: {prediction}")

    print("\n===================================")


# ============================================================
# 8. MAIN FUNCTION
# ============================================================

def main():

    customer_data = get_customer_input()

    churn_probability, prediction = predict_customer(
        customer_data
    )

    display_result(
        customer_data,
        churn_probability,
        prediction
    )


# ============================================================
# 9. EXECUTION
# ============================================================

if __name__ == "__main__":

    main()
