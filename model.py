import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib


X_train = pd.DataFrame()
y_train = pd.DataFrame()

X_test = pd.DataFrame()
y_test = pd.DataFrame()

mse_avg = 0
rmse_avg = 0


def linearRegression():
    model = LinearRegression()
    model.fit(X_train, y_train)
    joblib.dump(model, "linear_model.pkl")
    print("LinearRegression model is saved")

def elasticNet():
    model = ElasticNet()
    model.fit(X_train, y_train)
    joblib.dump(model, "elastic_model.pkl")
    print("ElasticNet model is saved")

def decisionTreeRegressor():
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    joblib.dump(model, "decisionTree_model.pkl")
    print("DecisionTreeRegressor model is saved")

def randomForestRegressor():
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    joblib.dump(model, "randomForest_model.pkl")
    print("RandomForestRegressor model is saved")

def gradientBoostingRegressor():
    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)
    joblib.dump(model, "gradientBoosting_model.pkl")
    print("GradientBoostingRegressor model is saved")

def kNeighborsRegressor():
    model = KNeighborsRegressor()
    model.fit(X_train, y_train)
    joblib.dump(model, "kNeighbors_model.pkl")
    print("KNeighborsRegressor model is saved")

def svr():
    model = SVR()
    model.fit(X_train, y_train)
    joblib.dump(model, "svr_model.pkl")
    print("SVR model is saved")

def CreateAndSaveModels():
    linearRegression()
    elasticNet()
    decisionTreeRegressor()
    randomForestRegressor()
    gradientBoostingRegressor()
    kNeighborsRegressor()
    svr()

def test(name):

    model = joblib.load(name + ".pkl")
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, y_pred)

    print(f'Mean Squared Error {name}: {mse:.2f}')
    print(f'Root Mean Squared Error {name}: {rmse:.2f}')
    print(f'Mean Absolute Error {name}: {mae:.2f}')

    if mse < mse_avg:
        print(f"The {name} is better than the simple average model.")
    else:
        print(f"The simple average model is better than the {name}.")

    print()

def TestModels():
    test("linear_model")
    test("elastic_model")
    test("decisionTree_model")
    test("randomForest_model")
    test("gradientBoosting_model")
    test("kNeighbors_model")
    test("svr_model")
