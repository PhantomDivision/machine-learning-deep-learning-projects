
import pandas as pd

# Sample insurance customer data
data = {
    "age": [25, 45, 32, 50, 28, 60, 35, 42, 29, 55],
    "premium": [5000, 12000, 7000, 15000, 6000,
                18000, 9000, 11000, 5500, 16000],
    "policy_tenure": [1, 10, 3, 12, 2, 15, 5, 8, 2, 14],
    "complaints": [3, 0, 2, 0, 4, 0, 1, 1, 3, 0],
    "payment_delay": [30, 0, 15, 0, 45, 0, 10, 5, 35, 0],
    "churn": [1, 0, 1, 0, 1, 0, 0, 0, 1, 0]
}

# Convert data into DataFrame
df = pd.DataFrame(data)

# Save dataset
df.to_csv("data/insurance_customers.csv", index=False)

print("Dataset created successfully!")
print("Total customers:", len(df))
print("Dataset location: data/insurance_customers.csv")
