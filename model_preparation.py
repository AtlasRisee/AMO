import pandas as pd
import model

if __name__ == "__main__":

    print("3. model_preparation.py\n")

    df_train = pd.read_csv("data/train/train_scaled.csv")

    # Create X_test and y_test
    model.X_train = df_train.drop('energy_consumption', axis=1)
    model.y_train = df_train['energy_consumption']

    model.CreateAndSaveModels()
