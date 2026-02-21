# ==========================================================
# Wine Quality Prediction Project
# Single File - VS Code Ready
# ==========================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ----------------------------------------------------------
# 0. Create outputs folder automatically
# ----------------------------------------------------------
os.makedirs("outputs", exist_ok=True)

# ----------------------------------------------------------
# 1. Load Dataset
# ----------------------------------------------------------
# Make sure winequality.csv is in same folder
df = pd.read_csv("winequality.csv")

print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== DATASET INFO =====")
print(df.info())

print("\n===== STATISTICS =====")
print(df.describe())

# ----------------------------------------------------------
# 2. Missing Values
# ----------------------------------------------------------
print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

# ----------------------------------------------------------
# 3. Handle Categorical Column (color)
# ----------------------------------------------------------
# Convert red/white -> numeric
if "color" in df.columns:
    df["color"] = df["color"].map({"red": 0, "white": 1})

# ----------------------------------------------------------
# 4. Exploratory Data Analysis (EDA)
# ----------------------------------------------------------

# ----- Quality Distribution -----
plt.figure(figsize=(6, 4))
sns.countplot(x="quality", data=df)
plt.title("Wine Quality Distribution")
plt.tight_layout()
plt.savefig("outputs/quality_distribution.png")
plt.show()

# ----- Correlation Heatmap (numeric only) -----
numeric_df = df.select_dtypes(include=["number"])

plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("outputs/correlation_heatmap.png")
plt.show()

# ----------------------------------------------------------
# 5. Prepare Data
# ----------------------------------------------------------
X = df.drop("quality", axis=1)
y = df["quality"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------------------------------------
# 6. Model Training
# ----------------------------------------------------------
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train_scaled, y_train)

# ----------------------------------------------------------
# 7. Prediction & Evaluation
# ----------------------------------------------------------
predictions = model.predict(X_test_scaled)

# R2 Score
r2 = r2_score(y_test, predictions)

# RMSE (compatible with all sklearn versions)
mse = mean_squared_error(y_test, predictions)
rmse = mse ** 0.5

print("\n===== MODEL PERFORMANCE =====")
print(f"R2 Score : {r2:.3f}")
print(f"RMSE     : {rmse:.3f}")

# ----------------------------------------------------------
# 8. Feature Importance
# ----------------------------------------------------------
importance = model.feature_importances_
feature_names = X.columns

feat_imp = pd.Series(importance, index=feature_names)
feat_imp = feat_imp.sort_values(ascending=True)

plt.figure(figsize=(8, 6))
feat_imp.plot(kind="barh")
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("outputs/feature_importance.png")
plt.show()

# ----------------------------------------------------------
# 9. Sample Prediction
# ----------------------------------------------------------
sample = X_test.iloc[0:1]
sample_scaled = scaler.transform(sample)

pred_quality = model.predict(sample_scaled)

print("\n===== SAMPLE PREDICTION =====")
print("Predicted Quality :", round(pred_quality[0], 2))
print("Actual Quality    :", y_test.iloc[0])

# ----------------------------------------------------------
# 10. End Message
# ----------------------------------------------------------
print("\n✅ Project executed successfully.")
print("📁 Graphs saved inside: outputs/")
