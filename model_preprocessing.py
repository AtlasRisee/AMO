import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def extract_date(df: pd.DataFrame):
    # Extract year, month, day, hour, minute, and second from the 'date' column
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    df['second'] = df['date'].dt.second
    return df.drop('date', axis=1)


def extract_scaled(df: pd.DataFrame, arr: np.ndarray):
    # Extract year, month, day, hour, minute, and second from the 'date' column
    df['year'] = arr[0]
    df['month'] = arr[1]
    df['day'] = arr[2]
    df['hour'] = arr[3]
    df['minute'] = arr[4]
    df['second'] = arr[5]
    return df.drop('date', axis=1)


if __name__ == "__main__":

    df_train = pd.read_csv('data/train/train.csv')
    df_test = pd.read_csv('data/test/test.csv')

    df_test['date'] = pd.to_datetime(df_test['date'])
    df_train['date'] = pd.to_datetime(df_train['date'])

    # Extract year, month, day, hour, minute, and second from the 'date' column
    X_test = extract_date(df_test)
    X_train = extract_date(df_train)

    X_train = X_train.drop('energy_consumption', axis=1)
    X_test = X_test.drop('energy_consumption', axis=1)

    scaler = StandardScaler()
    scaled_X_train = scaler.fit_transform(X_train)
    scaled_X_test = scaler.transform(X_test)

    df_test = extract_scaled(df_test, scaled_X_test.T)
    df_train = extract_scaled(df_train, scaled_X_train.T)

    df_train.to_csv('data/train/train_scaled.csv')
    df_test.to_csv('data/test/test_scaled.csv')
    df_train.info()
    df_test.info()
