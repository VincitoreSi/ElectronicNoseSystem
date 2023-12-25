"""
file: logistic.py
author: @VincitoreSi
date: 2023-12-16
brief: Logistic Regression classifier for the binary classification problem
"""

from dependencies.dependencies import *
from helper import *
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

def app_logistic(X_train, X_test, y_train, y_test, classes):
    """For Streamlit app"""
    logistic = LogisticModel(X_train, X_test, y_train, y_test, classes)
    logistic.run()
    logistic.save()
    st.pyplot(plt)
    st.write(logistic.results)
    st.markdown(
        get_download_link(f"output/models/classification/LogisticRegression.joblib"),
        unsafe_allow_html=True,
    )

def main():
    """The main function"""
    X_train, X_test, y_train, y_test = load_gas_data("Data/data/expanded_data.csv")
    classes = {1: "Ethylene", 2: "Acetone", 3: "Ethylene + Acetone"}
    logistic = LogisticModel(X_train, X_test, y_train, y_test, classes)
    logistic.run()
    logistic.save()

if __name__ == "__main__":
    main()
