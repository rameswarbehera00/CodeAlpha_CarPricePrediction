import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

os.makedirs("models", exist_ok=True)
os.makedirs("notebooks", exist_ok=True)

# 1. Load data
data_path = os.path.join("data", "car data.csv")
df = pd.read_csv(data_path)

# Normalize column names by trimming any accidental spaces
df.columns = df.columns.str.strip()
print("Normalized Column Names:", df.columns.tolist())

# 2. Feature Engineering: Derive Car_Age from Year
reference_year = 2020
df["Car_Age"] = reference_year - df["Year"]

# 3. Drop unneeded identifier columns (using errors='ignore' ensures safety)
df_clean = df.drop(columns=["Car_Name", "Year"], errors="ignore")

# 4. One-Hot Encoding for categorical features (drop_first=True prevents multicollinearity)
df_encoded = pd.get_dummies(df_clean, drop_first=True)

# 5. Split Features (X) and Target (y)
X = df_encoded.drop(columns=["Selling_Price"])
y = df_encoded["Selling_Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Train Models
# Baseline: Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)

# Ensemble: Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

# 7. Evaluate Metrics
def get_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return r2, mae, rmse

lr_r2, lr_mae, lr_rmse = get_metrics(y_test, lr_preds)
rf_r2, rf_mae, rf_rmse = get_metrics(y_test, rf_preds)

print("\n" + "=" * 55)
print("             MODEL EVALUATION COMPARISON            ")
print("=" * 55)
print(f"Linear Regression    -> R2: {lr_r2:.4f} | MAE: {lr_mae:.3f} | RMSE: {lr_rmse:.3f}")
print(f"Random Forest        -> R2: {rf_r2:.4f} | MAE: {rf_mae:.3f} | RMSE: {rf_rmse:.3f}")
print("=" * 55)

# 8. Feature Importance Plot (Random Forest)
plt.figure(figsize=(9, 5))
feat_importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
sns.barplot(x=feat_importances.values, y=feat_importances.index, palette="Blues_r")
plt.title("Random Forest Feature Importance", fontsize=13, fontweight="bold")
plt.xlabel("Relative Importance Score", fontsize=11)
plt.ylabel("Features", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join("notebooks", "feature_importance.png"), dpi=300)
plt.close()
print("\nSaved: notebooks/feature_importance.png")

# 9. Actual vs Predicted Price Plot
plt.figure(figsize=(7, 6))
plt.scatter(y_test, rf_preds, alpha=0.7, color="blue", edgecolors="k")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
plt.xlabel("Actual Selling Price (in Lakhs)", fontsize=11)
plt.ylabel("Predicted Selling Price (in Lakhs)", fontsize=11)
plt.title("Actual vs Predicted Car Prices (Random Forest)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join("notebooks", "actual_vs_predicted.png"), dpi=300)
plt.close()
print("Saved: notebooks/actual_vs_predicted.png")

# 10. Save the best model and column structure for inference
joblib.dump(rf, os.path.join("models", "car_price_rf_model.pkl"))
joblib.dump(X.columns.tolist(), os.path.join("models", "feature_columns.pkl"))
print("\nTrained model and feature columns saved in 'models/' directory.")