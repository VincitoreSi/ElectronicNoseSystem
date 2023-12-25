"""
file: elasticnet.py
author: @VincitoreSi
date: 2023-12-23
brief: ElasticNet model for predicting the concentration of the gases
"""

from dependencies.dependencies import *
from helper import plot_confusion_matrix, lead_and_prepare_data, load_gas_data, load_gas_data_for_regression
from .basemodel import BaseModel

class ElasticNetRegressionModel(BaseModel):
    """
    ElasticNet model for predicting the concentration of the gases
    :param X_train: training data
    :param X_test: testing data
    :param y_train: training labels
    :param y_test: testing labels
    :return: ElasticNet model
    """

    def __init__(self, X_train, X_test, y_train, y_test):
        super().__init__(
            "ElasticNet",
            ElasticNet(),
            X_train,
            X_test,
            y_train,
            y_test,
        )

def main():
    """The main function"""
    X_train, X_test, y_train, y_test = load_gas_data_for_regression("Data/data/expanded_data.csv", 2)
    shape = 1 if len(y_train.shape) == 1 else y_train.shape[1]
    if shape == 1:
        elasticnet_regression = ElasticNetRegressionModel(X_train, X_test, y_train, y_test)
        elasticnet_regression.train()
        elasticnet_regression.test()
    else:
        for i in range(shape):
            elasticnet_regression = ElasticNetRegressionModel(X_train, X_test, y_train.iloc[:, i], y_test.iloc[:, i])
            elasticnet_regression.train()
            elasticnet_regression.test()

if __name__ == "__main__":
    main()