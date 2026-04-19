import joblib
import sys
import pandas as pd
import numpy as np
import time
from xrfm import xRFM
import torch
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, roc_curve, auc
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

sizes = [5000, 7000, 9000, 11000, 13000, 15000, 17000]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random_state = 48
#
xgb_param = joblib.load("xgb_model.joblib")["best_param"]
mlp_param = joblib.load("mlp_model.joblib")["best_param"]
# rfm_param = joblib.load("xrfm_model.joblib")["best_param"]
rfm_param = {'model': {'kernel': 'lpq_kermac', 'bandwidth': np.float64(8.329253178564588), 'exponent': np.float64(1.2419210617329313), 'norm_p': 1.5520269069374861, 'diag': False, 'bandwidth_mode': 'constant'}, 'fit': {'reg': np.float64(0.002799976603973412), 'M_batch_size': 1000, 'iters': 5, 'early_stop_rfm': True, 'verbose': False}}
#
df_train = pd.read_csv("train.csv")
X_train, y_train = df_train.iloc[:, :-1], df_train.iloc[:, -1:]
df_val = pd.read_csv("validation.csv")
X_val, y_val = df_val.iloc[:, :-1], df_val.iloc[:, -1:]
df_test = pd.read_csv("test.csv")
X_test, y_test = df_test.iloc[:, :-1], df_test.iloc[:, -1:]

def model_performance(size, model_name, training_time, perf, X_train, y_train, X_test, y_test, X_val=None, y_val=None ):
    model = None
    if model_name == "xrfm":
        model = xRFM(
            rfm_params=rfm_param,
            random_state=random_state,
            device=device,
            tuning_metric="accuracy",
            split_method='top_vector_agop_on_subset',
            min_subset_size=10000
        )
        
        start = time.time()
        model.fit(X_train, y_train, X_val, y_val)
        end = time.time()
        
        y_pred = model.predict(X_test)
        
        training_time[model_name].append(end - start)
        perf[model_name].append(accuracy_score(y_test, y_pred))
        return
    
    elif model_name == "xgb":
        model = XGBClassifier(
            **xgb_param,
            objective="binary:logistic"
        )
    elif model_name == "mlp":
        model = MLPClassifier(
            **mlp_param,
            random_state=42, early_stopping=True, max_iter=500
        )
    else:
        sys.exit(1)
    
    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()
    
    y_pred = model.predict(X_test)
    
    training_time[model_name].append(end - start)
    perf[model_name].append(accuracy_score(y_test, y_pred))
    
training_time = {
    "mlp": [],
    "xgb": [],
    "xrfm": []
}
perf = {
    "mlp": [],
    "xgb": [],
    "xrfm": []
}
for size in sizes:
    X_train_s, y_train_s = resample(
        X_train, y_train, n_samples=size, random_state=random_state
    )
    X_val_s, y_val_s = resample(
        X_val, y_val, n_samples=size, random_state=random_state
    )
    X_test_s, y_test_s = resample(
        X_test, y_test, n_samples=size, random_state=random_state
    )
    
    X_train_np = X_train_s.to_numpy().astype(np.float32)
    X_val_np = X_val_s.to_numpy().astype(np.float32)
    y_train_np = y_train_s.to_numpy().astype(np.float32)
    y_val_np = y_val_s.to_numpy().astype(np.float32)
    X_test_np = X_test_s.to_numpy().astype(np.float32)
    y_test_np = y_test_s.to_numpy().astype(np.float32)
    
    
    model_performance(size, "xrfm", training_time, perf, X_train_np, y_train_np, X_test_np, y_test_np, X_val_np, y_val_np)
    model_performance(size, "xgb", training_time, perf, X_train_s, y_train_s, X_test_s, y_test_s    )
    model_performance(size, "mlp", training_time, perf, X_train_s, y_train_s.ravel(), X_test_s, y_test_s.ravel())
    
    
fig, axes = plt.subplots(2, 1, figsize=(10,8))

def plot_axes(ax, X, y, fmt, label):
    ax.plot(X, y, fmt=fmt, label=label)
    
mlp_fmt = "o-b"
xrfm_fmt = "^-r"
xgb_fmt = "s-g"

plot_axes(axes[0], sizes, training_time["xrfm"], xrfm_fmt, "xrfm")
plot_axes(axes[0], sizes, training_time["mlp"], mlp_fmt, "mlp")
plot_axes(axes[0], sizes, training_time["xgb"], xgb_fmt, "xgb")

axes[0].set_xlabel("Number of training sample")
axes[0].set_ylabel("Training Time (seconds)")
axes[0].set_title("Training Time Comparision")
axes[0].legend()
plot_axes(axes[1], sizes, perf["xrfm"], xrfm_fmt, "xrfm")
plot_axes(axes[1], sizes, perf["mlp"], mlp_fmt, "mlp")
plot_axes(axes[1], sizes, perf["xgb"], xgb_fmt, "xgb")

axes[1].set_xlabel("Number of training sample")
axes[1].set_ylabel("Accuracy")
axes[1].set_title("Performance Comparision")
axes[1].legend()

plt.tight_layout()
plt.show()
