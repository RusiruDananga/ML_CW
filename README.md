# Supermarket Sales and Quantity Prediction

## Overview

This project focuses on predicting the quantity sold and the net value of sales for a supermarket chain with two main stores: 'ABC' and 'XYZ'. The objective is to support sales and marketing decisions, manage suppliers, and optimize stock levels. The predictions are made using time series analysis and machine learning models.

## Methodology
Given the time series nature of the data, various regression models were tested to determine the best approach for prediction. The models evaluated include:

1. XGBoost Regressor
2. Linear Regression
3. Random Forest Regressor

The project pipeline includes Exploratory Data Analysis (EDA), model building, and model deployment. The entire process is version-controlled using GitHub, and the frontend for the model's predictions is implemented using [Streamlit](https://streamlit.io/).

## Data Preparation
### Data Sources

Two CSV files serve as the main data sources:

1. training_data.csv (614,098 records)
2. test_data.csv (247,624 records)

Columns in the dataset:

1. date_id: Date of the transaction
2. item_dept: Category of the item sold
3. item_qty: Quantity sold in the transaction
4. net_sales: Total revenue from the sale
5. store: Store where the sale occurred
6. item: Item identifier
7. invoice_num: Invoice identifier

### Data Cleaning and Feature Engineering
Removed the item and invoice_num columns since they don't contribute to the prediction.
Grouped the data by date_id, item_dept, and store to make the records unique.
Created lag features for item_qty and net_sales to improve model accuracy.

## Model Development
### Model Evaluation
Three models were trained and evaluated using metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), R-squared (R²), and Mean Absolute Percentage Error (MAPE). The best-performing model, XGBoost, was selected for deployment.

### Hyperparameter Optimization
Hyperparameters like the number of estimators, learning rate, maximum depth, subsample, alpha, and lambda were tuned to avoid overfitting and improve model performance.

### Feature Importance
Feature importance was analyzed to determine the impact of each feature on the prediction accuracy. XGBoost provides built-in feature importance metrics.

## Pipelines
A pipeline was created to automate the EDA, model building, and model saving processes. The pipeline ensures that the final datasets are prepared, and the models are trained and saved as pickle files for future use.

## Model Deployment
The models were integrated into a Streamlit web application for real-time predictions. The web app has four pages where users can input necessary values for prediction.

## Findings
1. Seasonal Trends: The analysis revealed significant sales spikes in December (Christmas) and January (New Year) for both stores, especially for store 'ABC'.
2. Feature Importance: Lag features and the item_dept column were identified as the most impactful features for prediction.
3. Model Performance: The XGBoost model outperformed Linear Regression and Random Forest, with the lowest MAPE and highest R² scores, making it the best model for this dataset.

## Conclusions
The project successfully built a model that accurately predicts both the quantity of items sold and the net sales for future dates. The findings can be leveraged to make informed sales, marketing, and stock management decisions. The integration with a user-friendly Streamlit application makes it easy to deploy the model and obtain predictions, ensuring its practical use in the supermarket's operations.

## Repository
You can find all the files, including the data, models, and scripts, in the GitHub repository below:

[GitHub Repository](https://github.com/RusiruDananga/ML_CW.git)

## Run the Application

```bash
streamlit run index.py
```

## Hosted URL
[https://mlcw-masters.streamlit.app/](https://mlcw-masters.streamlit.app/)