import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
import joblib
import numpy as np
import time

SEED = 42
df_train = pd.read_csv("train.csv")
X_train, y_train = df_train.iloc[:, :-1], df_train.iloc[:, -1:]

df_val = pd.read_csv("validation.csv")
X_val, y_val = df_val.iloc[:, :-1], df_val.iloc[:, -1]

X_train = X_train.to_numpy().astype(np.float32)
X_val = X_val.to_numpy().astype(np.float32)
y_train = y_train.to_numpy().astype(np.float32)
y_val = y_val.to_numpy().astype(np.float32)

cv = StratifiedKFold(n_splits=5)

model = XGBClassifier(
    random_state=SEED,
    objective="binary:logistic",
)

param_grid = {
    "max_depth": np.arange(2, 15, 1),
    "learning_rate": np.linspace(0.01, 2, 50),
    "n_estimators": [5, 100, 50]
}

search = RandomizedSearchCV(
    estimator=model, param_distributions=param_grid, scoring="accuracy", n_iter=200, cv=cv, verbose=True
)

search.fit(X_val, y_val)

final_model = XGBClassifier(
    **search.best_params_,
    random_state=SEED,
    objective="binary:logistic"
)

start = time.time()
final_model.fit(X_train, y_train)
end = time.time()

dump_info = {
    "model": final_model,
    "training_time": end - start,
    "best_param": search.best_params_
}
joblib.dump(dump_info, "xgb_model.joblib")
