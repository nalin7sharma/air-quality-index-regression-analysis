# ======================================================
# AIR QUALITY INDEX (AQI) - REGRESSION ANALYSIS PROJECT
# ======================================================

# -----------------------------
# 1. IMPORT REQUIRED LIBRARIES
# -----------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# 2. LOAD DATASET
# -----------------------------
print("\n========== LOADING DATASET ==========\n")

df = pd.read_csv("updated_pollution_dataset.csv")

print("First 5 rows of dataset:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

# Drop missing values (if any)
df = df.dropna()

# -------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS (EDA) — FIXED VERSION
# -------------------------------------------------

print("\n========== PERFORMING EDA ==========\n")

# Use only numeric columns for plots & correlation
numeric_df = df.select_dtypes(include=['number'])

# Feature distribution plots
numeric_df.hist(figsize=(10, 8))
plt.tight_layout()
plt.savefig("feature_distributions.png")
plt.close()

# Correlation heatmap (ONLY NUMERIC COLUMNS)
plt.figure(figsize=(8, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.close()

print("EDA plots saved as:")
print("- feature_distributions.png")
print("- correlation_heatmap.png")

# -------------------------------------------------
# 4. CREATE A NUMERIC TARGET (VERY IMPORTANT FIX)
# -------------------------------------------------
# Your dataset does NOT have AQI, it has "Air Quality" labels.
# So we will create a numeric AQI score from it.

def convert_air_quality_to_aqi(value):
    if value == "Good":
        return 50
    elif value == "Moderate":
        return 100
    elif value == "Poor":
        return 150
    else:
        return 100

df["AQI"] = df["Air Quality"].apply(convert_air_quality_to_aqi)

# -----------------------------
# 5. SIMPLE LINEAR REGRESSION
# -----------------------------
print("\n========== SIMPLE LINEAR REGRESSION ==========\n")

X = df[['PM2.5']]
y = df['AQI']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model_simple = LinearRegression()
model_simple.fit(X_train, y_train)

# Plot regression line
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, model_simple.predict(X_test), color='red')
plt.xlabel("PM2.5")
plt.ylabel("AQI")
plt.title("Simple Linear Regression: PM2.5 vs AQI")
plt.savefig("simple_linear_regression.png")
plt.close()

print("Simple Linear Regression Results:")
print("Slope:", model_simple.coef_[0])
print("Intercept:", model_simple.intercept_)

# -----------------------------
# 6. MULTIPLE LINEAR REGRESSION
# -----------------------------
print("\n========== MULTIPLE LINEAR REGRESSION ==========\n")

X = df[['PM2.5', 'PM10', 'NO2']]
y = df['AQI']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model_multi = LinearRegression()
model_multi.fit(X_train, y_train)

y_pred = model_multi.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Multiple Linear Regression Performance:")
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

print("\nFeature Coefficients:")
for col, coef in zip(X.columns, model_multi.coef_):
    print(f"{col}: {coef}")

# -----------------------------
# 7. POLYNOMIAL REGRESSION
# -----------------------------
print("\n========== POLYNOMIAL REGRESSION ==========\n")

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_poly, y, test_size=0.2, random_state=42
)

model_poly = LinearRegression()
model_poly.fit(X_train, y_train)

y_pred_poly = model_poly.predict(X_test)

print("Polynomial Regression Performance:")
print("MSE:", mean_squared_error(y_test, y_pred_poly))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_poly)))
print("R2 Score:", r2_score(y_test, y_pred_poly))

# -----------------------------
# 8. REGULARIZATION (RIDGE & LASSO)
# -----------------------------
print("\n========== REGULARIZATION ==========\n")

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
print("Ridge Coefficients:")
print(ridge.coef_)

lasso = Lasso(alpha=0.01)
lasso.fit(X_train, y_train)
print("\nLasso Coefficients:")
print(lasso.coef_)

# -----------------------------
# 9. MODEL DIAGNOSTICS (RESIDUAL PLOT)
# -----------------------------
print("\n========== MODEL DIAGNOSTICS ==========\n")

residuals = y_test - y_pred

plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='red')
plt.xlabel("Predicted AQI")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.savefig("residual_plot.png")
plt.close()

print("Residual plot saved as: residual_plot.png")

print("\n========== PROJECT COMPLETED SUCCESSFULLY ==========")
