import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

print("NYC Taxi Trip Duration Analysis")
print("=" * 50)

# 1. Data Loading and Initial Exploration
print("\n1. Data Loading and Initial Exploration")
print("-" * 40)
df_jan = pd.read_parquet('yellow_tripdata_2023-01.parquet')
print(f'Number of columns in January data: {len(df_jan.columns)}')
print('\nColumns:')
print(df_jan.columns.tolist())

# 2. Computing Duration
print("\n2. Computing Duration")
print("-" * 40)
df_jan['duration'] = (df_jan['tpep_dropoff_datetime'] - df_jan['tpep_pickup_datetime']).dt.total_seconds() / 60
duration_std = df_jan['duration'].std()
print(f'Standard deviation of trip durations: {duration_std:.2f} minutes')

# 3. Dropping Outliers
print("\n3. Dropping Outliers")
print("-" * 40)
df_jan_filtered = df_jan[(df_jan['duration'] >= 1) & (df_jan['duration'] <= 60)]
fraction_remaining = len(df_jan_filtered) / len(df_jan)
print(f'Fraction of records remaining after filtering: {fraction_remaining:.2%}')

# 4. One-Hot Encoding
print("\n4. One-Hot Encoding")
print("-" * 40)
df_jan_filtered['PULocationID'] = df_jan_filtered['PULocationID'].astype(str)
df_jan_filtered['DOLocationID'] = df_jan_filtered['DOLocationID'].astype(str)
feature_dicts = df_jan_filtered[['PULocationID', 'DOLocationID']].to_dict('records')
dv = DictVectorizer()
X = dv.fit_transform(feature_dicts)
print(f'Number of features after one-hot encoding: {X.shape[1]}')

# 5. Training a Model
print("\n5. Training a Model")
print("-" * 40)
y = df_jan_filtered['duration'].values
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
train_rmse = mean_squared_error(y, y_pred) ** 0.5
print(f'RMSE on training data: {train_rmse:.2f} minutes')

# 6. Model Evaluation
print("\n6. Model Evaluation")
print("-" * 40)
df_feb = pd.read_parquet('yellow_tripdata_2023-02.parquet')
df_feb['duration'] = (df_feb['tpep_dropoff_datetime'] - df_feb['tpep_pickup_datetime']).dt.total_seconds() / 60
df_feb = df_feb[(df_feb['duration'] >= 1) & (df_feb['duration'] <= 60)]
df_feb['PULocationID'] = df_feb['PULocationID'].astype(str)
df_feb['DOLocationID'] = df_feb['DOLocationID'].astype(str)
val_dicts = df_feb[['PULocationID', 'DOLocationID']].to_dict('records')
X_val = dv.transform(val_dicts)
y_val = df_feb['duration'].values
y_val_pred = model.predict(X_val)
val_rmse = mean_squared_error(y_val, y_val_pred) ** 0.5
print(f'RMSE on validation data: {val_rmse:.2f} minutes') 