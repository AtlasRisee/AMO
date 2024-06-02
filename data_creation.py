import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Constants
NUM_HOURS_IN_DAY = 24
NUM_DAYS_IN_YEAR = 365
MIN_ENERGY_CONSUMPTION = 100
MAX_ENERGY_CONSUMPTION = 500
TEST_SIZE = 0.1
RANDOM_STATE = 101

def generate_data(start_date, num_hours_in_day, num_days_in_year, min_energy_consumption, max_energy_consumption):
    num_hours = num_hours_in_day * num_days_in_year
    energy_consumption = np.random.randint(min_energy_consumption, max_energy_consumption, num_hours)
    date_range = pd.date_range(start=start_date, periods=num_hours, freq='H')
    data = {'date': date_range, 'energy_consumption': energy_consumption}
    return pd.DataFrame(data)

def split_data(df, test_size, random_state):
    X = df['date']
    y = df['energy_consumption']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def save_data(df, directory, filename):
    if not os.path.isdir(directory):
        os.makedirs(directory)
    df.to_csv(os.path.join(directory, filename), index=False)

if __name__ == "__main__":
    print("1. data_creation.py\n")

    df = generate_data('1/1/2024', NUM_HOURS_IN_DAY, NUM_DAYS_IN_YEAR, MIN_ENERGY_CONSUMPTION, MAX_ENERGY_CONSUMPTION)
    df.info()

    X_train, X_test, y_train, y_test = split_data(df, TEST_SIZE, RANDOM_STATE)
    print('len(X_train):', len(X_train))
    print('len(X_test):', len(X_test))

    df_train = pd.DataFrame({'date': X_train, 'energy_consumption': y_train})
    df_test = pd.DataFrame({'date': X_test, 'energy_consumption': y_test})

    save_data(df_train, 'data/train', 'train.csv')
    save_data(df_test, 'data/test', 'test.csv')
