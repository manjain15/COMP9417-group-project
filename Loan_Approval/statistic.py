import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
from xgboost import plot_importance
from xrfm.tree_utils import get_param_tree
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
import time
import math
import warnings
warnings.filterwarnings('ignore')

# Prevent output being printed out
import os
import sys
from contextlib import redirect_stdout

df_test = pd.read_csv("test.csv")
X_test, y_test = df_test.iloc[:, :-1], df_test.iloc[:, -1:]

X_test_np = X_test.to_numpy().astype(np.float32)
y_test_np = y_test.to_numpy().astype(np.float32)

best_models = {
    "xrfm": joblib.load("xrfm_model.joblib"),
    "xgb": joblib.load("xgb_model.joblib"),
    "mlp": joblib.load("mlp_model.joblib")
}

xfrm = joblib.load("xrfm_model.joblib")["model"]
xgb = joblib.load("xgb_model.joblib")["model"]
mlp = joblib.load("mlp_model.joblib")["model"]

# Display the accuracy, auc, training, inference time
result = {
    "Models": [],
    "Accuracy": [],
    "ROC-AUC": [],
    "Training Time": [],
    "Inference Time": [],
}

for name, metadata in best_models.items():
    model = metadata["model"]
    training_t = metadata["training_time"]
    
    start = time.time()
    y_pred = model.predict(X_test_np)
    y_pred_proba = model.predict_proba(X_test_np)[:, 1]
    end = time.time()
    fpr, tpr, threshold = roc_curve(y_test_np, y_pred_proba)
    auc_score = auc(fpr, tpr)
    
    result["Models"].append(name)
    result["Accuracy"].append(accuracy_score(y_test_np, y_pred))
    result["Inference Time"].append((end - start) / X_test.shape[0])
    result["Training Time"].append(training_t)
    result["ROC-AUC"].append(auc_score)
    
result_df = pd.DataFrame(result)
print(result_df)
print("Saving result table....")
result_df.to_csv("Basic_Statistic_Btw_Models.csv", index=False)

# ROC-AUC plot
def plot_roc_graph(model, label, color, X_test, y_test):
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, threshold = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, label=label + f" (AUC = {roc_auc:0.2f})", color=color)


plt.figure(0)
plt.title("ROC-AUC for different models")
plot_roc_graph(xfrm, "xRFM ROC", "red", X_test_np, y_test_np)
plot_roc_graph(xgb, "xGBoost ROC", "green", X_test, y_test)
plot_roc_graph(mlp, "MLP ROC", "blue", X_test, y_test)
plt.plot([0, 1], [0, 1], 'k--', label='Non predictive value')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(alpha=0.3)
plt.legend()
plt.savefig('ROC-AUC-plot.png', dpi=150, bbox_inches='tight')

# Feature Learning
# Display the AGOP for feature learning
def display_matrix(ax, matrix, color, leaf_count, tree_count):
    if len(matrix.shape) == 1 or matrix.shape[1] == 1:
        matrix = torch.diag(matrix)

    sns.heatmap(
        matrix,
        cmap=color,
        linewidths=0.5,
        ax=ax,
        alpha=0.8,
        linecolor="grey"
    )
    ax.set_title(f"AGOP of xRFM Leaf {leaf_count}, Tree {tree_count}")

def print_leaf_weight_rec(param_tree, tree_count, leaf_count, axes):
    if param_tree["type"] == "leaf":
        if param_tree["M"] == None:
            return leaf_count
        leaf_count += 1
        display_matrix(axes[leaf_count - 1], param_tree["M"], "Reds", leaf_count, tree_count)
        return leaf_count
    
    leaf_count = print_leaf_weight_rec(param_tree["left"], tree_count, leaf_count, axes)
    leaf_count = print_leaf_weight_rec(param_tree["right"], tree_count, leaf_count, axes)
    return leaf_count

tree_count = 0
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
for tree in xfrm.trees:
    tree_count += 1
    param_tree = get_param_tree(tree, is_root=True)
    print_leaf_weight_rec(param_tree, tree_count, 0, axes)
    
plt.savefig('AGOP_Heatmap.png', dpi=150, bbox_inches='tight')

# PCA Loadings
# Fit PCA
print("Evaluate PCA loadings...\n")
pca = PCA()
pca.fit(X_test_np)

# Calculate loading
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
loadings_df = pd.DataFrame(
    loadings, index=X_test.columns, columns=[f"PC{i + 1}" for i in range(len(X_test.columns))]
)

# Mutual Information Scores
print("Evaluate mutual information score...\n")
mi_scores = mutual_info_classif(X_test_np, y_test_np.ravel(), random_state=48)
mi_series = pd.Series(mi_scores, index=X_test.columns, name="MI score")

# Permutation importance
print("Evaluate permutation important.....\n")
def get_permutation_importance(model, name, X, y):
    scorer = None
    if name == "xRFM":
        def xrfm_scorer(estimator, X_inp, y_inp):
            y_pred = estimator.predict(X_inp)
            return accuracy_score(y_pred, y_inp)
        
        scorer = xrfm_scorer
    else:
        scorer = "accuracy"
      
    with open(os.devnull, 'w') as f:
        with redirect_stdout(f):  
            result = permutation_importance(
                model, X, y, scoring=scorer, n_repeats=5, random_state=48, n_jobs=1
            )

    perm_importance = pd.Series(result.importances_mean, index=X_test.columns, name=f"PI_{name}")
    return perm_importance

rfm_important = get_permutation_importance(xfrm, "xRFM", X_test_np, y_test_np)
xgb_important = get_permutation_importance(xgb, "XGBoost", X_test, y_test)
mlp_important = get_permutation_importance(mlp, "MLP", X_test, y_test)

# xgb_important, rfm_important, mlp_important
comparision_df = pd.concat([mi_series, loadings_df, xgb_important, rfm_important, mlp_important], axis=1)
display_comparision_df = pd.concat([mi_series, loadings_df["PC1"], xgb_important, rfm_important, mlp_important], axis=1)

display_comparision_df = display_comparision_df.sort_values(by="PI_XGBoost", ascending=False)

print(display_comparision_df)

comparision_df = comparision_df.sort_values(by="PI_XGBoost", ascending=False)
comparision_df.to_csv("feature_importance.csv", index=True)

# Plot the feature importance of the xGBoost
plot_importance(xgb)
plt.savefig('xGBoost_feature_important.png', dpi=150, bbox_inches='tight')

plt.tight_layout()
plt.show()