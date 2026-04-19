import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
import joblib
import numpy as np
import time
import torch

df_train = pd.read_csv("train.csv")
X_train, y_train = df_train.iloc[:20000, :-1], df_train.iloc[:20000, -1]

df_val = pd.read_csv("validation.csv")
X_val, y_val = df_val.iloc[:, :-1], df_val.iloc[:, -1]

cv = StratifiedKFold(n_splits=5)

model = MLPClassifier(random_state=42, early_stopping=True, max_iter=500)

param_grid = {
    "hidden_layer_sizes": [(50,), (50, 50), (100,)],
    "activation": ["relu", "tanh"],
    "solver": ["sgd", "adam"],
    "alpha": np.linspace(0.001, 2, 10),
    "batch_size": np.arange(200, 600, 50),
    "learning_rate": ["constant", "adaptive"],
    "learning_rate_init": np.linspace(0.001, 1, 10)
}

search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_grid,
    cv=cv,
    n_iter=100,
    scoring="accuracy",
    verbose=True,
    n_jobs=2
)

search.fit(X_val, y_val)

final_model = MLPClassifier(
    **search.best_params_,
    random_state=42, early_stopping=True, max_iter=500
)

start = time.time()
final_model.fit(X_train, y_train)
end = time.time()

dump_info = {
    "model": final_model,
    "training_time": end - start,
    "best_param": search.best_params_
}
joblib.dump(dump_info, "mlp_model.joblib")


