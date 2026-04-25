import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xrfm import xRFM

df = pd.read_csv('retail_store_sales.csv')
df = df.dropna(subset=['Total Spent'])
y = df['Total Spent'].to_numpy()

X_raw = df[['Price Per Unit', 'Quantity']]

X_train, X_test, y_train, y_test = train_test_split(X_raw, y, test_size=0.2, random_state=42)

print("Training XGBoost:")   
xgb_model = xgb.XGBRegressor(random_state=42)
xgb_model.fit(X_train, y_train)

print("Training xRFM:")
xrfm_model = xRFM()
xrfm_model.fit(X_train.to_numpy(), y_train, X_test.to_numpy(), y_test)