import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# -------------------------------------------------------------
# STEP 1: Load Dataset
# -------------------------------------------------------------
df = pd.read_csv("sales_data.csv")

# Convert date column to datetime
df["date"] = pd.to_datetime(df["date"])

# Sort by date (VERY important for time-based forecasting)
df = df.sort_values("date")

print("\nDataset Loaded Successfully!")
print(f"Total Rows: {len(df)}\n")

# -------------------------------------------------------------
# STEP 2: Feature Engineering
# -------------------------------------------------------------
df["day"] = df["date"].dt.day
df["month"] = df["date"].dt.month
df["year"] = df["date"].dt.year
df["day_of_week"] = df["date"].dt.dayofweek

# Features and target
feature_columns = [
    "day", "month", "year",
    "day_of_week", "promotions", "holiday_flag"
]

X = df[feature_columns]
y = df["sales"]

# -------------------------------------------------------------
# STEP 3: Time-Based Train-Test Split (80% Train, 20% Test)
# -------------------------------------------------------------
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print(f"Training rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}\n")

# -------------------------------------------------------------
# STEP 4: Train Model
# -------------------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

print("Model Trained Successfully!\n")

# -------------------------------------------------------------
# STEP 5: Evaluate Model
# -------------------------------------------------------------
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)

print("Model Evaluation:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"On average, predictions are off by ~{mae:.0f} sales units.\n")

# Show sample predictions
comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred.round(0)
})

print("Sample Predictions:")
print(comparison.head(10).to_string(index=False), "\n")

# -------------------------------------------------------------
# STEP 6: Predict Future Dates (Example: March 2024)
# -------------------------------------------------------------
future_data = pd.DataFrame({
    "day": [15, 20, 25],
    "month": [3, 3, 3],
    "year": [2024, 2024, 2024],
    "day_of_week": [4, 2, 0],   # Fri, Wed, Mon
    "promotions": [1, 0, 1],
    "holiday_flag": [0, 0, 0]
})

future_predictions = model.predict(future_data)

print("Future Sales Predictions:")
for i, pred in enumerate(future_predictions):
    row = future_data.iloc[i]
    print(
        f"2024-{row['month']:02d}-{row['day']:02d} | "
        f"Promotion: {'Yes' if row['promotions'] else 'No'} "
        f"→ Predicted Sales: {pred:.0f}"
    )

# -------------------------------------------------------------
# STEP 7: Plot Historical Sales
# -------------------------------------------------------------
plt.figure()
plt.plot(df["date"], df["sales"])
plt.title("Historical Sales Trend")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()