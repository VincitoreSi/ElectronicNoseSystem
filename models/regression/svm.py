"""
file: svm.py
author: @VincitoreSi
date: 2023-12-21
brief: SVM model for predicting the concentration of the gases
"""

from dependencies.dependencies import *
from helper import plot_confusion_matrix, lead_and_prepare_data, load_gas_data, load_gas_data_for_regression
from .basemodel import BaseModel

class SVMRegressionModel(BaseModel):
    """
    SVM model for predicting the concentration of the gases
    :param X_train: training data
    :param X_test: testing data
    :param y_train: training labels
    :param y_test: testing labels
    :return: SVM model
    """

    def __init__(self, X_train, X_test, y_train, y_test):
        super().__init__(
            "SVR",
            SVR(),
            X_train,
            X_test,
            y_train,
            y_test,
        )

def main():
    """The main function"""
    X_train, X_test, y_train, y_test = load_gas_data_for_regression("Data/data/expanded_data.csv", 1)
    svm_regression = SVMRegressionModel(X_train, X_test, y_train, y_test)
    svm_regression.train()
    svm_regression.test()

if __name__ == "__main__":
    main()