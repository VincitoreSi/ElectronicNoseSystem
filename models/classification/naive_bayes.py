"""
file: naive_bayes.py
author: @VincitoreSi
date: 2023-12-16
brief: Naive Bayes classifier for the binary classification problem
"""

from dependencies.dependencies import *
from helper import *
from .basemodel import BaseModel


class NaiveBayes(BaseModel):
    """
    Naive Bayes classifier for the binary classification problem
    :param X_train: training data
    :param X_test: testing data
    :param y_train: training labels
    :param y_test: testing labels
    :return: Naive Bayes classifier
    """

    def __init__(self, X_train, X_test, y_train, y_test, classes):
        super().__init__(
            "GaussianNB", GaussianNB(), X_train, X_test, y_train, y_test, classes
        )

def app_naive_bayes(X_train, X_test, y_train, y_test, classes):
    """For Streamlit app"""
    naive_bayes = NaiveBayes(X_train, X_test, y_train, y_test, classes)
    naive_bayes.run()
    naive_bayes.save()
    st.pyplot(plt)
    st.write(naive_bayes.results)
    st.markdown(
        get_download_link(f"output/models/classification/GaussianNB.joblib"),
        unsafe_allow_html=True,
    )

def main():
    """The main function"""
    X_train, X_test, y_train, y_test = load_gas_data("Data/data/expanded_data.csv")
    classes = {1: "Ethylene", 2: "Acetone", 3: "Ethylene + Acetone"}
    naive_bayes = NaiveBayes(X_train, X_test, y_train, y_test, classes)
    naive_bayes.run()
    naive_bayes.save()

if __name__ == "__main__":
    main()
