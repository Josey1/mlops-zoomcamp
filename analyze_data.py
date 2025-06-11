import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
import mlflow

# Question 3: Count records
df = pd.read_parquet('yellow_tripdata_2023-03.parquet')
print(f"Question 3 - Number of records: {len(df)}")

# Question 4: Data preparation
def read_dataframe(df):
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    return df

df_processed = read_dataframe(df)
print(f"Question 4 - Size after preparation: {len(df_processed)}")

# Question 5: Train model
categorical = ['PULocationID', 'DOLocationID']
numerical = ['trip_distance']

train_dicts = df_processed[categorical + numerical].to_dict(orient='records')
dv = DictVectorizer()
X_train = dv.fit_transform(train_dicts)
y_train = df_processed['duration'].values

model = LinearRegression()
model.fit(X_train, y_train)
print(f"Question 5 - Model intercept: {model.intercept_}")

# Question 6: Model size
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("nyc-taxi-experiment")

with mlflow.start_run():
    mlflow.sklearn.log_model(model, "model")
    model_info = mlflow.sklearn.log_model(model, "model")
    print(f"Question 6 - Model size: {model_info.model_size_bytes}") 