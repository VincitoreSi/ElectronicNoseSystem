"""
file: adaboost.py
author: @VincitoreSi
date: 2023-12-16
brief: AdaBoost classifier for the binary classification problem
"""

from dependencies.dependencies import *
from helper import *
from .basemodel import BaseModel


class AdaBoost(BaseModel):
    """
    AdaBoost classifier for the binary classification problem
    """

    def __init__(
        self, X_train, X_test, y_train, y_test, n_estimators, learning_rate, classes
    ):
        """
        Constructor
        :param X_train: training data
        :param X_test: testing data
        :param y_train: training labels
        :param y_test: testing labels
        :param n_estimators: number of estimators
        :param learning_rate: learning rate
        :param max_depth: maximum depth
        """
        super().__init__(
            "AdaBoostClassifier",
            AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate),
            X_train,
            X_test,
            y_train,
            y_test,
            classes,
        )

def app_adaboost(X_train, X_test, y_train, y_test, classes):
    """For Streamlit app"""
    adaboost = AdaBoost(X_train, X_test, y_train, y_test, 100, 1, classes)
    adaboost.run()
    adaboost.save()
    st.pyplot(plt)
    st.write(adaboost.results)
    st.markdown(
        get_download_link(f"output/models/classification/AdaBoostClassifier.joblib"),
        unsafe_allow_html=True,
    )

def main():
    """
    Main function
    """
    X_train, X_test, y_train, y_test = load_gas_data("Data/data/expanded_data.csv")
    classes = {1: "Ethylene", 2: "Acetone", 3: "Ethylene + Acetone"}
    adaboost = AdaBoost(X_train, X_test, y_train, y_test, 100, 1, classes)
    adaboost.run()
    adaboost.save()


if __name__ == "__main__":
    main()
