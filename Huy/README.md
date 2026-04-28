# Retail Store Sales Dataset - COMP9417 Group A

## Overview
This folder contains the preprocessing, modeling, and scalability stress testing for the **Retail Store Sales** dataset. It serves as our robustness and scalability baseline for evaluating xRFM against XGBoost and MLP on messy tabular data.

**Task:** Regression  
**Target Variable:** `Total Spent`  
**Key Challenge:** Handling missing values, categorical encoding, and evaluating the exponential computational bottlenecks of kernel machines.

## Files in this Directory
* `retail_store_sales.csv` - The raw synthetic sales dataset containing missing and inconsistent values.
* `retail.ipynb` - The main Jupyter Notebook containing data imputation, model training, and the subsample stress test loop.
* `retail_rmse.png` - Learning curve plot showing Test RMSE vs. Sample Size.
* `retail_stress_test.png` - Computational plot showing Training Time vs. Sample Size.

## Methodology Summary
1. **Preprocessing:** Median imputation for numerical features, most-frequent imputation and ordinal encoding for categorical features. Standard scaling applied.
2. **Models Trained:** xRFM, XGBoost, and MLP.
3. **Evaluation Metrics:** RMSE (Root Mean Squared Error) and Training Time (Seconds).

## How to Run
1. Ensure the required packages are installed (`pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, and `xRFM`).
2. Open `retail.ipynb` in Jupyter Notebook or Jupyter Lab.
3. Run all cells sequentially to reproduce the data cleaning, the $n=1000$ to $n=9576$ scalability stress test, and regenerate the final plots.

*(Note: xRFM training will take approximately 24 seconds on the full $n=9576$ sample size due to dense $O(n^3)$ kernel operations. XGBoost and MLP will train almost instantly).*
