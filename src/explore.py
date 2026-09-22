import os
import pandas as pd

# Load dataset
data_path = os.path.join("data", "car data.csv")
df = pd.read_csv(data_path)

print("--- DATASET OVERVIEW ---")
print(f"Total Rows: {df.shape[0]}, Total Columns: {df.shape[1]}")
print("\nFirst 5 Records:")
print(df.head())

print("\n--- DATA TYPES & NULL VALUES ---")
print(df.info())
print("\nMissing values per column:\n", df.isnull().sum())

print("\n--- CATEGORICAL FEATURE DISTRIBUTIONS ---")
print("Fuel Types:\n", df["Fuel_Type"].value_counts())
print("\nSelling Types:\n", df["Selling_type"].value_counts())
print("\nTransmission Types:\n", df["Transmission"].value_counts())
print("\nOwner Distribution:\n", df["Owner"].value_counts())

print("\n--- NUMERICAL STATISTICAL SUMMARY ---")
print(df.describe())