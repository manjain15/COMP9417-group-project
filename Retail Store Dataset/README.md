# Retail Store Sales Dataset — COMP9417 Project
## xRFM vs XGBoost vs MLP on Regression

This notebook reproduces all experiments for the Retail Store Sales dataset contribution to the COMP9417 Group Project. It evaluates three models (xRFM, XGBoost, MLP) on a regression task, establishing a robustness and scalability baseline for handling messy tabular data and computational bottlenecks.

---

## Requirements

Python 3.10 is required. All experiments were run using a virtual environment. To set one up:

bash
python3 -m venv comp9417_env
source comp9417_env/bin/activate
pip install pandas numpy scikit-learn xgboost matplotlib xrfm ipykernel
python -m ipykernel install --user --name=comp9417_env --display-name "comp9417"


**Mac users:** XGBoost requires OpenMP. Install it before running:

bash
brew install libomp


---

## How to Run

1. Activate the virtual environment:
   bash
   source comp9417_env/bin/activate
   

2. Launch Jupyter:
   bash
   jupyter notebook
   

3. Open `retail.ipynb` and select the `comp9417` kernel via Kernel → Change Kernel → comp9417.

4. Run all cells top to bottom using Kernel → Restart & Run All.

*Note: The dataset `retail_store_sales.csv` must be located in the same directory as the notebook. The scalability stress test will take roughly 45-60 seconds to complete due to xRFM's dense kernel operations at higher sample sizes.*

---

## Notebook Structure

| Cell | Description |
|------|-------------|
| 1 | Imports and setup. |
| 2 | Loads the `retail_store_sales.csv` dataset and drops rows with missing target values (`Total Spent`). |
| 3 | Preprocessing pipeline: applies median imputation and standard scaling to numerical features, and most-frequent imputation and ordinal encoding to categorical features. |
| 4 | Performs an 80/20 train/test split setting `random_state=42` for reproducibility. |
| 5 | Scalability stress test: loops through sample sizes n in {1000, 3000, 5000, 8000, 9576}. Trains xRFM, XGBoost, and MLP at each step and records training times and RMSE. |
| 6 | Plots Training Time vs. Sample Size. |
| 7 | Plots Test Performance (RMSE) vs. Sample Size. |
| 8 | Inference calculation: records the time required to predict on the test set and converts it to microseconds (us) per sample. |
| 9 | *Optional:* Dirty data test block demonstrating xRFM's fatal error on un-imputed missing values compared to XGBoost's native handling. |

---

## Outputs

Running the notebook produces the following visualization files (which are referenced in Appendix B.1 of the final report):

| File | Description |
|------|-------------|
| `retail_rmse.png` | Learning curve plot showing Test RMSE vs. Sample Size for all three models. |
| `retail_stress_test.png` | Computational curve showing Training Time (seconds) vs. Sample Size for all three models. |

---

## Reproducibility

All models and data splits use `random_state=42` for exact reproducibility. Unlike datasets fetched via API, this notebook relies on the local `retail_store_sales.csv` file. Results may vary slightly across machines due to CPU/threading differences during timing calculations, but predictive metrics (RMSE) should remain identical.

---

## Key Results (at Max Sample Size n=9576)

| Model | RMSE | Train Time (s) | Inference (us/sample) |
|-------|---------|----------------|----------------------|
| xRFM | 17.9581 | 23.6786 | 20.7903 |
| XGBoost | 16.8682 | 0.3257 | 0.8434 |
| MLP | 15.8754 | 18.0545 | 0.8343 |
