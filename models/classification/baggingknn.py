"""
file: baggingknn.py
author: @VincitoreSi
date: 2023-12-16
brief: Bagging classifier with KNN
"""

from dependencies.dependencies import *
from helper import plot_confusion_matrix, lead_and_prepare_data, load_gas_data
from .basemodel import BaseModel


class BaggingKNN(BaseModel):
    """
    Bagging classifier with KNN
    """

    def __init__(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        classes,
        n_estimators,
        max_samples,
        max_features,
    ):
        """
        :param X_train: training data
        :param X_test: testing data
        :param y_train: training labels
        :param y_test: testing labels
        :param classes: classes
        :param n_estimators: number of estimators
        :param max_samples: maximum number of samples
        :param max_features: maximum number of features
        """
        super().__init__(
            "BaggingKNN",
            BaggingClassifier(
                KNeighborsClassifier(n_neighbors=14),
                n_estimators=n_estimators,
                max_samples=max_samples,
                max_features=max_features,
            ),
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
    baggingknn = BaggingKNN(X_train, X_test, y_train, y_test, classes, 100, 0.5, 0.5)
    baggingknn.run()
    baggingknn.save()

if __name__ == "__main__":
    main()
