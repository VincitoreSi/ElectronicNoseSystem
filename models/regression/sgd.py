"""
file: sgd.py
author: @VincitoreSi
date: 2023-12-23
brief: SGD model for predicting the concentration of the gases
"""

from dependencies.dependencies import *
from helper import plot_confusion_matrix, lead_and_prepare_data, load_gas_data, load_gas_data_for_regression
from .basemodel import BaseModel

class SGDRegressionModel(BaseModel):
    """
    SGD model for predicting the concentration of the gases
    :param X_train: training data
    :param X_test: testing data
    :param y_train: training labels
    :param y_test: testing labels
    :return: SGD model
    """

    def __init__(self, X_train, X_test, y_train, y_test):
        super().__init__(
            "SGDRegressor",
            SGDRegressor(),
            X_train,
            X_test,
            y_train,
            y_test,
        )

def main():
    """The main function"""
    X_train, X_test, y_train, y_test = load_gas_data_for_regression("Data/data/expanded_data.csv", 1)
    shape = 1 if len(y_train.shape) == 1 else y_train.shape[1]
    if shape == 1:
        sgd_regression = SGDRegressionModel(X_train, X_test, y_train, y_test)
        sgd_regression.train()
        sgd_regression.test()
    else:
        for i in range(shape):
            sgd_regression = SGDRegressionModel(X_train, X_test, y_train.iloc[:, i], y_test.iloc[:, i])
            sgd_regression.train()
            sgd_regression.test()

if __name__ == "__main__":
    main()