import pandas as pd
from xrfm import xRFM
import torch
import joblib
import numpy as np
import time
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split

def loguniform(low, high, size):
    log_low = np.log10(low)
    log_high = np.log10(high)
    
    sample_log = np.random.uniform(log_low, log_high, size=size)
    
    samples = 10 ** sample_log
    
    return samples

def generate_rfm_params(size):
    bandwidths = loguniform(1, 200, size)
    bandwidth_modes = ["constant"]
    diagonals = [False, True]
    exponents = np.random.uniform(0.7, 1.4, size=size)  
    regularisations = loguniform(1e-6, 1, size=size)
    
    rfm_params_list = []
    
    for bandwidth, mode, diag, exp, reg in zip(bandwidths, bandwidth_modes, diagonals, exponents, regularisations):
        kernel = None
        p = np.random.uniform(0, 1)
        norm_p = None
        if p > 0.2:
            kernel = "lpq_kermac"
            norm_p = np.random.uniform(exp, exp + 0.8 * (2 - exp))
        else:
            kernel = "laplace"
        
        rfm_params_list.append(
            {
                "model": {
                    "kernel": kernel,
                    "bandwidth": bandwidth,
                    "exponent": exp,
                    "norm_p": norm_p,
                    "diag": diag,
                    "bandwidth_mode": mode
                },
                "fit": {
                    "reg": reg,
                    "iters": 1,
                    "early_stop_rfm": True,
                    "verbose": False
                }
            }
        )
    return rfm_params_list

class xRFMWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, device, rfm_params=None):
        self.rfm_params = rfm_params
        self.device = device
        self.random_state = 48
        self.model = None
        
    def fit(self, X, y):
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        self.model = xRFM(
            rfm_params=self.rfm_params,
            random_state=48,
            device=device,
            tuning_metric="accuracy",
            split_method='top_vector_agop_on_subset',
            min_subset_size=10000
        )
        self.model.fit(X_train, y_train, X_val, y_val)
        return self
    
    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
     
df_train = pd.read_csv("train.csv")
X_train, y_train = df_train.iloc[:, :-1], df_train.iloc[:, -1:]

df_val = pd.read_csv("validation.csv")
X_val, y_val = df_val.iloc[:, :-1], df_val.iloc[:, -1:]

# Convert to numpy
X_train = X_train.to_numpy().astype(np.float32)
X_val = X_val.to_numpy().astype(np.float32)
y_train = y_train.to_numpy().astype(np.float32)
y_val = y_val.to_numpy().astype(np.float32)

# Tuning hyperparameter
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = xRFMWrapper(device)
cv = StratifiedKFold(n_splits=5)
param_grid = {
    "rfm_params": generate_rfm_params(3)
}

search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=cv,
    scoring="accuracy",
    verbose=True,
    n_jobs=1
)
search.fit(X_val, y_val)
# End of tuning

rfm_params = search.best_params_["rfm_params"]
print(rfm_params)

model = xRFM(
    rfm_params=rfm_params,
    random_state=48,
    device=device,
    tuning_metric="accuracy",
    split_method='top_vector_agop_on_subset',
    min_subset_size=10000
)

start = time.time()
model.fit(X_train, y_train, X_val, y_val)
end = time.time()

dump_info = {
    "model": model,
    "training_time": end - start,
    "best_param": rfm_params
}
joblib.dump(dump_info, "xrfm_model.joblib")