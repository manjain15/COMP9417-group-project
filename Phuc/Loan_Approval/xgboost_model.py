import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import joblib
import numpy as np
import time

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
    objective="binary:logistic"
)

param_grid = {
    "max_depth": np.arange(2, 7, 1),
    "learning_rate": np.linspace(0.01, 1, 20),
    "n_estimators": [5, 100, 20]
}
search = GridSearchCV(
    estimator=model, param_grid=param_grid, scoring="accuracy", cv=cv, verbose=True
)

search.fit(X_val, y_val)

final_model = XGBClassifier(
    **search.best_params_,
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
