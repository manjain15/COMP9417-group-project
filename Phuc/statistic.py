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
# import os
# # Create a folder where you have lots of space and point to it
# os.environ['JOBLIB_TEMP_FOLDER'] = 'D:/temp_joblib'

df_test = pd.read_csv("test.csv")
X_test, y_test = df_test.iloc[:, :-1], df_test.iloc[:, -1:]

X_test_np = X_test.to_numpy().astype(np.float32)
y_test_np = y_test.to_numpy().astype(np.float32)

xfrm = joblib.load("xrfm_model.joblib")["model"]
xgb = joblib.load("xgb_model.joblib")["model"]
mlp = joblib.load("mlp_model.joblib")["model"]

# Plot the feature importance of the xGBoost
plt.figure(6)
plot_importance(xgb)

# Print accuracy and ROC-AUC plot
def print_accuracy(model, name, X_test, y_test):
    y_pred = model.predict(X_test)
    print(f"Accuracy of {name}: {accuracy_score(y_test, y_pred)}")
    
def plot_roc_graph(model, label, color, X_test, y_test):
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, threshold = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, label=label + f" (AUC = {roc_auc:0.2f})", color=color)

print_accuracy(xfrm, "xRFM Model", X_test_np, y_test_np)
print_accuracy(xgb, "xGBoost Model", X_test_np, y_test_np)
print_accuracy(mlp, "MLP Model", X_test, y_test)
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
plt.legend()

# Display the AGOP for feature learning
def display_matrix(matrix, color, leaf_count, tree_count):
    plt.figure(leaf_count)
    if len(matrix.shape) == 1 or matrix.shape[1] == 1:
        matrix = torch.diag(matrix)
        
    class_name = X_test.iloc[:, :-1].columns
    ax = sns.heatmap(
        matrix,
        cmap=color,
        yticklabels=class_name,
        xticklabels=class_name,
        linewidths=0.5,
        linecolor="grey"
    )
    ax.set_title(f"AGOP of xRFM Leaf {leaf_count}, Tree {tree_count}")

def print_leaf_weight_rec(param_tree, tree_count, leaf_count):
    if param_tree["type"] == "leaf":
        leaf_count += 1
        if param_tree["M"] == None:
            return leaf_count
        display_matrix(param_tree["M"], "Reds", leaf_count, tree_count)
        return leaf_count
    
    leaf_count = print_leaf_weight_rec(param_tree["left"], tree_count, leaf_count)
    leaf_count = print_leaf_weight_rec(param_tree["right"], tree_count, leaf_count)
    return leaf_count

tree_count = 0
for tree in xfrm.trees:
    tree_count += 1
    param_tree = get_param_tree(tree, is_root=True)
    print_leaf_weight_rec(param_tree, tree_count, 0)


# Feature Learning

# PCA Loadings
# Fit PCA
pca = PCA()
pca.fit(X_test_np)

# Calculate loading
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
loadings_df = pd.DataFrame(
    loadings, index=X_test.columns, columns=[f"PC{i + 1}" for i in range(len(X_test.columns))]
)

# Mutual Information Scores
mi_scores = mutual_info_classif(X_test_np, y_test_np.ravel(), random_state=48)
mi_series = pd.Series(mi_scores, index=X_test.columns, name="MI score")

# Permutation importance
def get_permutation_importance(model, name, X, y):
    scorer = None
    if name == "xRFM":
        def xrfm_scorer(estimator, X_inp, y_inp):
            y_pred = estimator.predict(X_inp)
            return accuracy_score(y_pred, y_inp)
        
        scorer = xrfm_scorer
    else:
        scorer = "accuracy"
        
    result = permutation_importance(
        model, X, y, scoring=scorer, n_repeats=5, random_state=48, n_jobs=1
    )

    perm_importance = pd.Series(result.importances_mean, index=X_test.columns, name=f"PI_{name}")
    return perm_importance

rfm_important = get_permutation_importance(xfrm, "xRFM", X_test_np, y_test_np)
xgb_important = get_permutation_importance(xgb, "XGBoost", X_test, y_test)
mlp_important = get_permutation_importance(mlp, "MLP", X_test, y_test)

# xgb_important, rfm_important, mlp_important
comparision_df = pd.concat([mi_series, loadings_df], axis=1)

comparision_df.sort_values(by="MI score", ascending=False)

print(comparision_df)
comparision_df.to_csv("feature_importance.csv", index=True)

plt.show()