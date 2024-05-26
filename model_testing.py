import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error


if __name__ == "__main__":

    print("4. model_testing.py\n")

    # Load model and pd.DataFrame
    model = joblib.load("model.pkl")
    df_test = pd.read_csv("data/test/test_scaled.csv")

    # Create X_test and y_test
    X_test = df_test.drop('energy_consumption', axis=1)
    y_test = df_test['energy_consumption']

    # Predict
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, y_pred)

    print(f'Mean Squared Error LinearRegression: {mse:.2f}')
    print(f'Root Mean Squared Error LinearRegression: {rmse:.2f}')
    print(f'Mean Absolute Error LinearRegression: {mae:.2f}')

    df_train = pd.read_csv("data/train/train_scaled.csv")

    y_train = df_train['energy_consumption']

    mse_avg = mean_squared_error(y_test, [y_train.mean()] * len(y_test))
    rmse_avg = mse_avg ** 0.5

    print(f'Mean Squared Error Average model: {mse_avg:.2f}')
    print(f'Root Mean Squared Error Average model: {rmse_avg:.2f}')

    if mse < mse_avg:
        print("The linear regression model is better than the simple average model.")
    else:
        print("The simple average model is better than the linear regression model.")
