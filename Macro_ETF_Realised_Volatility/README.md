# Macro ETF Realised Volatility Dataset — COMP9417 Project
## xRFM vs XGBoost vs MLP on Regression

This notebook reproduces all experiments for the Macro ETF Realised Volatility dataset contribution to the COMP9417 Group Project. It evaluates three models (xRFM, XGBoost, MLP) on a regression task — forecasting the forward 5-day realised volatility of six macro ETFs (SPY, QQQ, IWM, EFA, TLT, GLD) — establishing a robustness and scalability baseline on time-series financial data with hyperparameter tuning and interpretability analysis.

---

## Requirements

Python 3.10 is required. Set up an isolated Python environment using your preferred tool (`venv`, `conda`, `uv`, etc.), activate it, and install the following packages:

bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn scipy torch yfinance pyarrow xrfm ipykernel


Then register the environment as a Jupyter kernel so the notebook can find it:

bash
python -m ipykernel install --user --name=<your-env-name> --display-name "<display name>"


*Note: A live internet connection is required on first run to download OHLCV data via the `yfinance` API. Subsequent runs will read from the local `./cache/` parquet directory. xRFM will automatically use a CUDA GPU if available; otherwise it falls back to CPU.*

---

## How to Run

1. Set up a Python 3.10 environment with the dependencies listed in **Requirements** above (any virtual environment manager is fine — `venv`, `conda`, `uv`, etc.).

2. Launch Jupyter from that environment:
   bash
   jupyter notebook
   

3. Open `macro_realised_vol.ipynb` and select the kernel that points to the environment you just set up.

4. Run all cells top to bottom using Kernel → Restart & Run All.

*Note: The notebook fetches data via `yfinance` (no local CSV required) and caches it under `./cache/` as parquet. The hyperparameter tuning step (RandomizedSearchCV with n_iter=10) and the scalability stress test are the dominant runtime costs — expect 5-10 minutes end-to-end on CPU, less on GPU.*

---

## Notebook Structure

| Cell | Description |
|------|-------------|
| 1 | Imports, constants, and global setup. Sets `SEED=42`, `TARGET_WINDOW=5` (forward realised-vol horizon in trading days), the six tickers, the 2004-2024 date window, and the seven engineered feature names. |
| 2 | Defines `SklearnCompatibleXRFM`, a minimal scikit-learn wrapper around the xRFM regressor (handles 2D target padding, internal chronological validation split, and CUDA/CPU device selection). |
| 3 | Loads OHLCV data via `yfinance` with parquet caching, then engineers seven features per ticker: realised volatility (5d, 21d), momentum (5d, 21d), moving-average distance (20d, 50d), and a 5d volume-surge indicator. The target is the next 5 trading days' realised volatility, scaled by `TARGET_SCALE=10000.0` for log readability. |
| 4 | Date-based chronological 60/20/20 train/val/test split with a `TARGET_WINDOW`-day buffer at each edge to prevent target leakage. Applies `StandardScaler` fitted on train only. |
| 5 | Builds the tuning subset (last 8,000 rows of train + start of val) as a scikit-learn `PredefinedSplit`. Defines `MODEL_SPECS` containing hyperparameter grids for xRFM, XGBoost, and MLP, plus a `build_model` helper. |
| 6 | Runs `RandomizedSearchCV` (n_iter=10, scoring=neg_RMSE) for each model and stores the best params per model. |
| 7 | Refits each tuned model on train + val combined, evaluates on test, and records nRMSE, training time, and per-sample inference time. |
| 8 | Interpretability analysis: computes mutual information, PCA principal-component-1 loadings, XGBoost permutation importance, and the xRFM AGOP diagonal (averaged across all leaf models in all trees). |
| 9 | Scalability stress test: loops through sample sizes N in {2500, 5000, 10000, full train} and refits all three models from scratch at each step, recording training time and test nRMSE. |
| 10 | Plots Training Time vs. Sample Size (log scale) and Test nRMSE vs. Sample Size side-by-side. |
| 11 | Feature-importance comparison: normalises all four importance scores to [0, 1], renders a 2x2 bar-chart panel, computes Spearman rank correlations between the methods, and renders the correlation matrix as a heatmap. |

---

## Outputs

Running the notebook produces the following visualization file (which is referenced in Appendix B of the final report):

| File | Description |
|------|-------------|
| `interpretability_comparison.png` | 2x2 panel comparing the top features ranked by AGOP, Mutual Information, PCA Loading, and XGBoost Permutation Importance. |

The training-time vs. sample-size, nRMSE vs. sample-size, and Spearman rank-correlation heatmap plots are rendered inline in the notebook but are not saved to disk by default.

---

## Reproducibility

All models, data splits, and search procedures use `random_state=42` for exact reproducibility. The OHLCV data is pulled from `yfinance` and cached as parquet under `./cache/` on first run, so subsequent runs are deterministic and offline. Results may vary slightly across machines due to CPU/GPU threading differences during timing measurements, but predictive metrics (nRMSE) should remain identical given the same `yfinance` data snapshot.

---

## Key Results (Final Test Set, Train + Val Combined)

| Model | nRMSE | Train Time (s) | Inference (us/sample) |
|-------|---------|----------------|----------------------|
| xRFM | 0.7128 | 249.24 | 51.0 |
| XGBoost | 0.7309 | 0.18 | 1.0 |
| MLP | 0.7276 | 48.05 | 3.0 |
