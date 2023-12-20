"""
file: decisiontree.py
author: @VincitoreSi
date: 2023-12-16
brief: Decision Tree classifier for the binary classification problem
"""

from dependencies.dependencies import *
from helper import plot_confusion_matrix, lead_and_prepare_data, load_gas_data
from .basemodel import BaseModel


class DecisionTree(BaseModel):
    """
    Decision Tree classifier for the binary classification problem
    :param X_train: training data
    :param X_test: testing data
    :param y_train: training labels
    :param y_test: testing labels
    :param max_depth: maximum depth
    :return: Decision Tree classifier
    """

    def __init__(self, X_train, X_test, y_train, y_test, max_depth, classes):
        super().__init__(
            "DecisionTreeClassifier",
            DecisionTreeClassifier(max_depth=max_depth),
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
    decision_tree = DecisionTree(X_train, X_test, y_train, y_test, 10, classes)
    decision_tree.train()
    decision_tree.test()


if __name__ == "__main__":
    main()
