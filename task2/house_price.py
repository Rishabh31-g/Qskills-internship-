
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load dataset
df = pd.read_csv("train.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# 2. Remove ID column and separate target
df = df.drop(columns=["Id"], errors="ignore")
X = df.drop(columns=["SalePrice"])
y = df["SalePrice"]

# 3. Identify numerical and categorical columns
numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("Numerical features:", len(numeric_features))
print("Categorical features:", len(categorical_features))

# 4. Preprocess numerical and categorical data
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

# 5. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining houses:", len(X_train))
print("Testing houses:", len(X_test))

# 6. Create and train Linear Regression model
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

model.fit(X_train, y_train)
print("\nModel training completed!")

# 7. Make predictions
predictions = model.predict(X_test)

# 8. Evaluate the model
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\n----- MODEL EVALUATION -----")
print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
print(f"R2 Score: {r2:.4f} ({r2 * 100:.2f}%)")

# 9. Compare actual and predicted prices
results = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": predictions.round(2)
})

print("\n----- SAMPLE PREDICTIONS -----")
print(results.head(10).to_string(index=False))

# 10. Plot actual vs predicted prices
plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions, alpha=0.6)
plt.xlabel("Actual House Price ($)")
plt.ylabel("Predicted House Price ($)")
plt.title("Actual vs Predicted House Prices")

min_price = min(y_test.min(), predictions.min())
max_price = max(y_test.max(), predictions.max())
plt.plot(
    [min_price, max_price],
    [min_price, max_price],
    "r--"
)

plt.tight_layout()
plt.savefig("actual_vs_predicted.png")
plt.show()

# 11. Plot the most important numerical correlations
correlation_data = df[
    numeric_features + ["SalePrice"]
].corr(numeric_only=True)

top_features = (
    correlation_data["SalePrice"]
    .drop("SalePrice")
    .abs()
    .sort_values(ascending=False)
    .head(10)
    .index.tolist()
)

plt.figure(figsize=(10, 7))
sns.heatmap(
    correlation_data.loc[
        top_features + ["SalePrice"],
        top_features + ["SalePrice"]
    ],
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Top Numerical Features: Correlation with SalePrice")
plt.tight_layout()
plt.savefig("price_correlation_heatmap.png")
plt.show()

# 12. Save trained model
joblib.dump(model, "house_price_model.pkl")
print("\nModel saved as house_price_model.pkl")
print("Analysis completed successfully!")
