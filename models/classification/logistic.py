"""
file: logistic.py
author: @VincitoreSi
date: 2023-12-16
brief: Logistic Regression classifier for the binary classification problem
"""

from dependencies.dependencies import *
from helper import plot_confusion_matrix, lead_and_prepare_data, load_gas_data
from .basemodel import BaseModel


class LogisticModel(BaseModel):
    """
    Logistic Regression classifier for the binary classification problem
    :param X_train: training data
    :param X_test: testing data
    :param y_train: training labels
    :param y_test: testing labels
    :return: Logistic Regression classifier
    """

    def __init__(self, X_train, X_test, y_train, y_test, classes):
        super().__init__(
            "LogisticRegression",
            LogisticRegression(),
            X_train,
            X_test,
            y_train,
            y_test,
            classes,
        )


def main():
    """The main function"""
    X_train, X_test, y_train, y_test = load_gas_data("Data/data/expanded_data.csv")
    classes = {1: "Ethylene", 2: "Acetone", 3: "Ethylene + Acetone"}
    logistic = LogisticModel(X_train, X_test, y_train, y_test, classes)
    logistic.train()
    logistic.test()


if __name__ == "__main__":
    main()
