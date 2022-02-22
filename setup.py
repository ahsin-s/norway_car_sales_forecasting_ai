import io

import pandas as pd
import numpy as np

import requests

from config import Config

from sklearn import svm
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from skgarden.quantile import RandomForestQuantileRegressor, ExtraTreesQuantileRegressor


def load_csv_from_url(config: Config) -> pd.DataFrame:
    resp = requests.get(config.data_config.data_url)
    if resp.status_code == 200:
        content = resp.content.decode('utf-8')
        csv_content = io.StringIO(content)
        df = pd.read_csv(csv_content)
        assert df.shape[1] > 1
        return df

    raise Exception("Failed to retrieve data: " + str(resp))


def preprocess(data: pd.DataFrame) -> pd.DataFrame:
    # Create a column "Period" with both the Year and the Month
    data["Period"] = data["Year"].astype(str) + "-" + data["Month"].astype(str)

    # We use the datetime formatting to make sure format is consistent
    data["Period"] = pd.to_datetime(data["Period"]).dt.strftime(" % Y - % m")

    # Create a pivot of the data to show the periods on columns and the car makers on rows 
    data = pd.pivot_table(data=data, values="Quantity", index="Make", columns="Period", aggfunc="sum", fill_value=0)

    return data


def train_test_split(df,
                     len_input=12,
                     len_forecast=1,
                     test_len=12):
    """
    domain specific method to split the data into train/test subsets
    len_input: the size of the input to the model. This corresponds to the number of
            quarters the model sees when learning the relationship between input/output
    len_forecast: the number of quarters out the model forecasts.
            1 = model forecasts 1 quarter ahead. 5 = model forecasts 5 quarters
    test_len: how many quarters are clipped from the df to hold out for testing
    """
    # if test_len < len_input:
    #     raise ValueError("test_len must be greater than or equal to len_input")
    D = df.values

    periods = D.shape[1]

    # Training set creation: run through all the possible time windows
    loops = periods + 1 - len_input - len_forecast - test_len
    train = [D[:, col:col + len_input + len_forecast] for col in range(loops)]
    train = np.vstack(train)
    X_train, Y_train = np.split(train, [len_input], axis=1)

    # Test set creation: unseen “future” data with the demand just before
    test = [D[:, col:col + len_input + len_forecast] for col in range(loops, loops + test_len)]
    test = np.vstack(test)
    X_test, Y_test = np.split(test, [len_input], axis=1)

    # this data formatting is needed if we only predict a single period
    if len_forecast == 1:
        Y_train = Y_train.ravel()  # unravel a multidimensional array into a 1-d dimension
        Y_test = Y_test.ravel()

    return X_train, Y_train, X_test, Y_test


def train_regression_tree(X_train, Y_train):
    from sklearn.tree import DecisionTreeRegressor

    # — Instantiate a Decision Tree Regressor
    tree = DecisionTreeRegressor(max_depth=5, min_samples_leaf=5)
    # — Fit the tree to the training data
    tree.fit(X_train, Y_train)
    return tree


def train_random_forest_regressor(X_train, Y_train):
    from skgarden.quantile import RandomForestQuantileRegressor

    rf_quantile_reg = RandomForestQuantileRegressor(max_depth=5, min_samples_leaf=5)
    rf_quantile_reg.fit(X_train, Y_train)
    return rf_quantile_reg


def predict(model, X):
    return model.predict(X)


def evaluate_predictions(predictions: np.ndarray, ground_truth: np.ndarray):
    """
    MAE calculation
    """
    return np.mean(abs(ground_truth - predictions)) / np.mean(ground_truth)


def train_and_evaluate(model, X_train, Y_train, X_test, Y_test):
    model.fit(X_train, Y_train)
    predictions = predict(model, X_test)
    performance = evaluate_predictions(predictions, Y_test)
    print(f"{model.__class__.__name__} performance: {performance}")
    return model, predictions, performance


def main():
    config = Config()

    # setup
    df = load_csv_from_url(config)
    df = preprocess(df)
    xtrain, ytrain, xtest, ytest = train_test_split(df)

    # tree
    tree = DecisionTreeRegressor(**config.random_forest_config.hyperparameters)
    tree_model, tree_predictions, tree_performance = train_and_evaluate(tree, xtrain, ytrain, xtest, ytest)

    # random forest
    rf_model = RandomForestQuantileRegressor(max_depth=5, min_samples_leaf=5)
    rf_model, rf_predictions, rf_performance = train_and_evaluate(rf_model, xtrain, ytrain, xtest, ytest)

    # extra trees
    et_model = ExtraTreesQuantileRegressor(min_samples_leaf=5, max_depth=5)
    et_model, et_predictions, et_performance = train_and_evaluate(et_model, xtrain, ytrain, xtest, ytest)

    # svr
    svr_model = svm.SVR()
    svr_model, svr_predictions, svr_performance = train_and_evaluate(svr_model, xtrain, ytrain, xtest, ytest)

    # adaboost
    adaboost_model = AdaBoostRegressor(n_estimators=100)
    adaboost_model, adaboost_predictions, adaboost_performance = train_and_evaluate(adaboost_model, xtrain, ytrain, xtest, ytest)


if __name__ == "__main__":
    main()
