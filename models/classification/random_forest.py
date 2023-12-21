"""
file: random_forest.py
author: @VincitoreSi
date: 2023-12-16
brief: Random Forest classifier for the binary classification problem
"""

from dependencies.dependencies import *
from helper import plot_confusion_matrix, lead_and_prepare_data, load_gas_data
from .basemodel import BaseModel


class RandomForest(BaseModel):
    """
    Random Forest classifier for the binary classification problem
    :param X_train: training data
    :param X_test: testing data
    :param y_train: training labels
    :param y_test: testing labels
    :param n_estimators: number of trees
    :return: Random Forest classifier
    """

    def __init__(self, X_train, X_test, y_train, y_test, n_estimators, classes):
        super().__init__(
            "RandomForestClassifier",
            RandomForestClassifier(n_estimators=n_estimators),
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
    random_forest = RandomForest(X_train, X_test, y_train, y_test, 100, classes)
    random_forest.train()
    random_forest.test()


if __name__ == "__main__":
    main()
