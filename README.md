# 🚗 Car Price Prediction with Machine Learning

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning regression project developed as part of the **CodeAlpha Data Science Internship (Task 3)**. The goal of this project is to accurately estimate the secondary market valuation (selling price in Lakhs) of used cars based on historical attributes, wear metrics, and vehicle specifications.

---

## 📌 Project Overview

Predicting vehicle depreciation and resale value involves analyzing multiple non-linear factors such as current showroom price, cumulative mileage, fuel configuration, transmission type, and ownership history. 

This project explores the dataset, engineers key chronological features, addresses multicollinearity through dummy encoding, and benchmarks a baseline **Linear Regression** model against an ensemble **Random Forest Regressor**.

---

## 📂 Repository Structure

```text
CodeAlpha_CarPricePrediction/
├── data/
│   └── car data.csv
├── models/
│   ├── car_price_rf_model.pkl
│   └── feature_columns.pkl
├── notebooks/
│   ├── Car_Price_Prediction.ipynb
│   ├── feature_importance.png
│   └── actual_vs_predicted.png
├── src/
│   ├── explore.py
│   ├── train.py
│   └── predict.py
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md

```

---

## ⚙️ Data Preprocessing & Feature Engineering

1. **Header Normalization**: Stripped leading and trailing whitespace across all dataset columns to prevent indexing exceptions.
2. **Car Age Derivation**: Subtracted the manufacturing year from the reference baseline ($2020 - \text{Year}$) to create a continuous depreciation metric (`Car_Age`).
3. **Carduality & Leakage Control**: Dropped nominal identifiers (`Car_Name`) to avoid overfitting on sparse labels.
4. **Categorical Encoding**: Applied one-hot encoding (`pd.get_dummies`) with `drop_first=True` on `Fuel_Type`, `Selling_type`, and `Transmission` to prevent the dummy variable trap (multicollinearity).

---

## 📊 Model Benchmarks & Evaluation

The dataset was split into **80% training** and **20% testing** subsets (`random_state=42`). Performance was evaluated using $R^2$ Score, Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE):

| Model | $R^2$ Score | MAE (Lakhs) | RMSE (Lakhs) |
| --- | --- | --- | --- |
| **Linear Regression (Baseline)** | 0.8489 | 1.216 | 1.866 |
| **Random Forest Regressor (Ensemble)** | **0.9585** | **0.649** | **0.977** |

The **Random Forest Regressor** captures non-linear interactions across vehicle specifications, explaining **~95.85%** of the price variance on unseen test data.

---

## 📈 Key Insights & Feature Importance

* **Present Price**: Serves as the primary anchor for market valuation.
* **Car Age**: Demonstrates the strongest negative correlation with resale value due to depreciation.
* **Fuel & Transmission**: Diesel and automatic variants command higher retention value in secondary markets.

---

## 🚀 Setup & Usage Instructions

### 1. Clone the Repository

```cmd
git clone [https://github.com/](https://github.com/)<your-username>/CodeAlpha_CarPricePrediction.git
cd CodeAlpha_CarPricePrediction

```

### 2. Set Up Virtual Environment & Dependencies

```cmd
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

```

### 3. Train Models & Export Visualizations

```cmd
python src\train.py

```

### 4. Run Interactive Inference

```cmd
python src\predict.py
