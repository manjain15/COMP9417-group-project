# Adult Income Dataset — COMP9417 Project
## xRFM vs XGBoost vs MLP on Binary Classification

This notebook reproduces all experiments for the Adult Income dataset contribution to the COMP9417 Group Project. It evaluates three models (xRFM, XGBoost, MLP) on a binary classification task and includes an interpretability comparison and scalability analysis.

---

## Requirements

Python 3.10 is required. All experiments were run using a virtual environment. To set one up:

```bash
python3 -m venv comp9417_env
source comp9417_env/bin/activate
pip install ucimlrepo xrfm xgboost scikit-learn matplotlib seaborn scipy ipykernel
python -m ipykernel install --user --name=comp9417_env --display-name "comp9417"
```

**Mac users:** XGBoost requires OpenMP. Install it before running:

```bash
brew install libomp
```

---

## How to Run

1. Activate the virtual environment:
   ```bash
   source comp9417_env/bin/activate
   ```

2. Launch Jupyter:
   ```bash
   jupyter notebook
   ```

3. Open `Adult-Income-Dataset.ipynb` and select the `comp9417` kernel via Kernel → Change Kernel → comp9417.

4. Run all cells top to bottom using Kernel → Restart & Run All.

The notebook will automatically download the Adult Income dataset from the UCI ML Repository. An internet connection is required for the first run.

---

## Notebook Structure

| Cell | Description |
|------|-------------|
| 1 | Imports and setup. Sets `SEED = 42` for reproducibility. |
| 2 | Loads the Adult Income dataset (UCI ID 2) via `fetch_ucirepo`. |
| 3 | Preprocessing: cleans target labels, drops `education` and `fnlwgt`, replaces missing values with `Unknown`, one-hot encodes categorical features (12 to 91 dimensions), and z-score normalises numerical features. |
| 4 | Stratified 60/20/20 train/validation/test split (29,305 / 9,768 / 9,769 samples). |
| 5 | Trains xRFM (`classification=True`, `random_state=42`). Records training time and per-sample inference time. |
| 6 | Tunes XGBoost via random search over 20 configurations. Best config: `n_estimators=500, max_depth=5, learning_rate=0.05, subsample=1.0`. |
| 7 | Tunes MLP over learning rates `{0.0001, 0.001, 0.01}` with early stopping. Best: `lr=0.001`. |
| 8 | Interpretability comparison: computes AGOP diagonal, PCA loadings, mutual information, and permutation importance. Normalises all scores to [0,1] and computes Spearman rank correlations between all method pairs. |
| 9 | Scalability analysis: trains XGBoost and MLP (and xRFM if memory allows) at subsample sizes of 5%, 10%, 20%, 40%, 60%, 80%, and 100% of the training set. Plots AUC-ROC and training time vs n. |
| 10 | Final results summary. Prints and saves all metrics to `adult_income_results.csv`. |

---

## Outputs

Running the notebook produces the following files in the same directory:

| File | Description |
|------|-------------|
| `adult_income_results.csv` | AUC-ROC, accuracy, training time, and inference time for all three models. |
| `interpretability_comparison.png` | Bar charts of top 10 features per importance method. |
| `rank_correlation_heatmap.png` | Spearman rank correlation heatmap between all four importance methods. |
| `scalability_plot.png` | AUC-ROC and training time vs training set size for all models. |

---

## Reproducibility

All random seeds are fixed at `SEED = 42` throughout. The dataset is downloaded programmatically from the UCI ML Repository so no manual data download is required. Results may vary slightly across machines due to floating point differences, but should be consistent within the same environment.

---

## Key Results

| Model | AUC-ROC | Accuracy | Train Time (s) | Inference (ms/sample) |
|-------|---------|----------|----------------|----------------------|
| xRFM | 0.8705 | 0.8384 | 160.24 | 0.0799 |
| XGBoost | 0.9310 | 0.8780 | 1.09 | 0.0027 |
| MLP | 0.9128 | 0.8589 | 59.15 | 0.0052 |
