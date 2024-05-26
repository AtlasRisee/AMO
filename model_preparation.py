import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression


if __name__ == "__main__":

    print("3. model_preparation.py\n")

    df_train = pd.read_csv("data/train/train_scaled.csv")

    # Create X_test and y_test
    X_train = df_train.drop('energy_consumption', axis=1)
    y_train = df_train['energy_consumption']

    model = LinearRegression()
    model.fit(X_train, y_train)
    joblib.dump(model, "model.pkl")
