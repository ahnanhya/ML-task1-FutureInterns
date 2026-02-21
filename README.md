# ML-task1-FutureInterns
# Sales Forecasting Using Linear Regression

## Overview
This project implements a basic machine learning model to forecast future sales using historical business data. The model is built using Linear Regression and includes time-based feature engineering, model evaluation, and visualization through a simple Flask web application.
The objective of this project is to demonstrate a complete machine learning workflow, from data preprocessing to prediction and visualization.

---
## Project Structure
```
FUTURE_ML_01/
│
├── sales_data.csv
├── sales_model.py
└── app.py
```
* `sales_data.csv` – Historical sales dataset
* `sales_model.py` – Script for training and evaluating the model
* `app.py` – Flask application to display evaluation results and graphs
---
## Features
* Date preprocessing and time-based feature extraction
* Linear Regression model training
* Time-based train-test split (80% training, 20% testing)
* Model evaluation using Mean Absolute Error (MAE)
* Visualization of:
  * Historical sales trend
  * Actual vs predicted sales
---
## Technologies Used
* Python
* Pandas
* Scikit-learn
* Matplotlib
* Flask
---
## Installation
Install the required dependencies:
```
pip install pandas scikit-learn matplotlib flask
```
---
## How to Run
1. Ensure all files are in the same project folder.
2. Run the Flask application:
```
python app.py
```
3. Open a browser and navigate to:
```
http://127.0.0.1:5000
```
The dashboard will display the model evaluation metric and the generated graphs.

---
## Model Evaluation
The model is evaluated using Mean Absolute Error (MAE).
MAE measures the average difference between actual and predicted sales values. A lower MAE indicates better prediction accuracy.

---
## Conclusion
This project demonstrates the implementation of a simple sales forecasting system using basic machine learning techniques. It covers data preprocessing, model training, evaluation, and visualization in a structured and reproducible manner.
