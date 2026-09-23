import os
import joblib
import pandas as pd

# Load saved artifacts
model_path = os.path.join("models", "car_price_rf_model.pkl")
columns_path = os.path.join("models", "feature_columns.pkl")

model = joblib.load(model_path)
feature_columns = joblib.load(columns_path)


def predict_car_price(
    present_price,
    driven_kms,
    fuel_type,
    selling_type,
    transmission,
    owner,
    car_age,
):
    """Predicts car selling price in lakhs given user inputs."""
    # 1. Construct single-row DataFrame
    raw_data = {
        "Present_Price": [present_price],
        "Driven_kms": [driven_kms],
        "Fuel_Type": [fuel_type],
        "Selling_type": [selling_type],
        "Transmission": [transmission],
        "Owner": [owner],
        "Car_Age": [car_age],
    }
    input_df = pd.DataFrame(raw_data)

    # 2. Apply One-Hot Encoding
    input_encoded = pd.get_dummies(input_df)

    # 3. Align with training columns (filling missing dummy columns with 0)
    aligned_df = input_encoded.reindex(columns=feature_columns, fill_value=0)

    # 4. Predict
    prediction = model.predict(aligned_df)[0]
    return max(0.0, prediction)  # Ensure non-negative valuation


if __name__ == "__main__":
    print("=" * 60)
    print("           CAR PRICE PREDICTION ENGINE (RANDOM FOREST)       ")
    print("=" * 60)

    # Sample Car 1: 2017 Honda City (Petrol, Dealer, Manual, 25,000 km, showroom ~10.5 Lakh)
    sample_age = 2020 - 2017
    price_est = predict_car_price(
        present_price=10.5,
        driven_kms=25000,
        fuel_type="Petrol",
        selling_type="Dealer",
        transmission="Manual",
        owner=0,
        car_age=sample_age,
    )
    print(f"Sample 1 (2017 Petrol Sedan, 25k km, 10.5L original):")
    print(f"  -> Estimated Resale Price: ₹ {price_est:.2f} Lakhs\n")

    # Sample Car 2: 2012 Diesel SUV (Individual, Manual, 95,000 km, showroom ~14.0 Lakh)
    sample_age_2 = 2020 - 2012
    price_est_2 = predict_car_price(
        present_price=14.0,
        driven_kms=95000,
        fuel_type="Diesel",
        selling_type="Individual",
        transmission="Manual",
        owner=1,
        car_age=sample_age_2,
    )
    print(f"Sample 2 (2012 Diesel SUV, 95k km, 14.0L original, 1 prev owner):")
    print(f"  -> Estimated Resale Price: ₹ {price_est_2:.2f} Lakhs")
    print("=" * 60)