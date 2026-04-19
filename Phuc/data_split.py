import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Loan Approval Prediction.csv")

print(f"The size of the data is: {df.shape}")

X_data = df.iloc[:, :-1]
Y_data = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X_data, Y_data, test_size=0.3, stratify=Y_data, random_state=48
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.3, stratify=y_train, random_state=48
)

num_cols = [
    "person_age","person_income","person_emp_exp","loan_amnt",
    "loan_int_rate","loan_percent_income","cb_person_cred_hist_length","credit_score"
]

scaler = StandardScaler()
X_num_train = pd.DataFrame(scaler.fit_transform(X_train[num_cols]), columns=num_cols, index=X_train.index)
X_num_val = pd.DataFrame(scaler.transform(X_val[num_cols]), columns=num_cols, index=X_val.index)
X_num_test = pd.DataFrame(scaler.transform(X_test[num_cols]), columns=num_cols, index=X_test.index)

X_rest_train = X_train[[col for col in X_train.columns if col not in num_cols]]
X_rest_val = X_val[[col for col in X_val.columns if col not in num_cols]]
X_rest_test = X_test[[col for col in X_test.columns if col not in num_cols]]


train_data = pd.concat([X_num_train, X_rest_train, y_train], axis=1)
val_data = pd.concat([X_num_val, X_rest_val, y_val], axis=1)
test_data = pd.concat([X_num_test, X_rest_test, y_test], axis=1)

train_data.to_csv("train.csv", index=False)
val_data.to_csv("validation.csv", index=False)
test_data.to_csv("test.csv", index=False)



