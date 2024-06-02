import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

# Constants
DATE_COMPONENTS = ['year', 'month', 'day', 'hour', 'minute', 'second']

def read_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    return pd.read_csv(file_path)

def extract_date_components(df):
    df[DATE_COMPONENTS] = df['date'].dt.to_period('S').dt.components
    return df.drop('date', axis=1)

def scale_data(df, scaler):
    return scaler.fit_transform(df)

def save_data(df, file_path):
    df.to_csv(file_path, index=False)

if __name__ == "__main__":
    print("2. model_preprocessing.py\n")

    df_train = read_data('data/train/train.csv')
    df_test = read_data('data/test/test.csv')

    df_test['date'] = pd.to_datetime(df_test['date'])
    df_train['date'] = pd.to_datetime(df_train['date'])

    X_train = extract_date_components(df_train)
    X_test = extract_date_components(df_test)

    X_train = X_train.drop('energy_consumption', axis=1)
    X_test = X_test.drop('energy_consumption', axis=1)

    scaler = StandardScaler()
    scaled_X_train = scale_data(X_train, scaler)
    scaled_X_test = scaler.transform(X_test)

    df_train[DATE_COMPONENTS] = scaled_X_train
    df_test[DATE_COMPONENTS] = scaled_X_test

    save_data(df_train, 'data/train/train_scaled.csv')
    save_data(df_test, 'data/test/test_scaled.csv')

    df_train.info()
    df_test.info()
