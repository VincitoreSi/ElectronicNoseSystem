"""
file: linear.py
author: @VincitoreSi
date: 2023-12-21
brief: Linear regression model for predicting the concentration of the gases
"""

from dependencies.dependencies import *
from helper import plot_confusion_matrix, lead_and_prepare_data, load_gas_data, load_gas_data_for_regression
from .basemodel import BaseModel

class LinearRegressionModel(BaseModel):
    """
    Linear regression model for predicting the concentration of the gases
    :param X_train: training data
    :param X_test: testing data
    :param y_train: training labels
    :param y_test: testing labels
    :return: Linear regression model
    """

    def __init__(self, X_train, X_test, y_train, y_test):
        super().__init__(
            "LinearRegression",
            LinearRegression(),
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
        linear_regression = LinearRegressionModel(X_train, X_test, y_train, y_test)
        linear_regression.train()
        linear_regression.test()
    else:
        for i in range(shape):
            linear_regression = LinearRegressionModel(X_train, X_test, y_train.iloc[:, i], y_test.iloc[:, i])
            linear_regression.train()
            linear_regression.test()


if __name__ == "__main__":
    main()