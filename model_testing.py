from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import model


if __name__ == "__main__":

    print("4. model_testing.py\n")

    # Load model and pd.DataFrame
    df_test = pd.read_csv("data/test/test_scaled.csv")
    df_train = pd.read_csv("data/train/train_scaled.csv")

    # Create X_test and y_test
    model.X_test = df_test.drop('energy_consumption', axis=1)
    model.y_test = df_test['energy_consumption']

    # Create X_test and y_test
    model.y_train = df_train['energy_consumption']

    model.mse_avg = mean_squared_error(model.y_test, [model.y_train.mean()] * len(model.y_test))
    model.rmse_avg = model.mse_avg ** 0.5

    print(f'Mean Squared Error Average model: {model.mse_avg:.2f}')
    print(f'Root Mean Squared Error Average model: {model.rmse_avg:.2f}')
    print()

    model.TestModels()
