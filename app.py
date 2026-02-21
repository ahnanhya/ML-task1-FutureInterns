# =============================================================
# Simple Flask App for Sales Forecast Visualization
# =============================================================
# Requirements:
#   pip install flask pandas scikit-learn matplotlib
# =============================================================

from flask import Flask, render_template_string
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import os

app = Flask(__name__)

# -------------------------------------------------------------
# TRAIN MODEL WHEN SERVER STARTS
# -------------------------------------------------------------
df = pd.read_csv("sales_data.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

df["day"] = df["date"].dt.day
df["month"] = df["date"].dt.month
df["year"] = df["date"].dt.year
df["day_of_week"] = df["date"].dt.dayofweek

feature_columns = [
    "day", "month", "year",
    "day_of_week", "promotions", "holiday_flag"
]

X = df[feature_columns]
y = df["sales"]

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

print("Model trained and ready!")

# -------------------------------------------------------------
# ROUTE: HOME
# -------------------------------------------------------------
@app.route("/")
def home():

    # -------- Graph 1: Historical Sales --------
    plt.figure()
    plt.plot(df["date"], df["sales"])
    plt.title("Historical Sales Trend")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.xticks(rotation=45)
    plt.tight_layout()

    historical_path = "static/historical.png"
    os.makedirs("static", exist_ok=True)
    plt.savefig(historical_path)
    plt.close()

    # -------- Graph 2: Actual vs Predicted --------
    plt.figure()
    plt.plot(df["date"].iloc[split_index:], y_test.values, label="Actual")
    plt.plot(df["date"].iloc[split_index:], y_pred, label="Predicted")
    plt.title("Actual vs Predicted Sales")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    prediction_path = "static/prediction.png"
    plt.savefig(prediction_path)
    plt.close()

    html = f"""
    <h1>Sales Forecast Dashboard</h1>
    <p><strong>Model MAE:</strong> {mae:.2f}</p>

    <h2>Historical Sales</h2>
    <img src="/static/historical.png" width="800">

    <h2>Actual vs Predicted (Test Data)</h2>
    <img src="/static/prediction.png" width="800">
    """

    return render_template_string(html)


if __name__ == "__main__":
    app.run(debug=True)