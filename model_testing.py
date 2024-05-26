import joblib
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score


if __name__ == "__main__":

    # Load model and pd.DataFrame
    model = joblib.load("model.pkl")
    df_test = pd.read_csv("data/test/test_scaled.csv")

    # Create X_test and y_test
    X_test = df_test.drop('energy_consumption', axis=1)
    y_test = df_test['energy_consumption']

    # Predict
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Print results
    print("Accuracy on test data:", accuracy)
    print(classification_report(y_test, y_pred))
