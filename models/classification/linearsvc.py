"""
file: linearsvc.py
author: @VincitoreSi
date: 2023-12-16
brief: LinearSVC classifier for the binary classification problem
"""

from dependencies.dependencies import *
from helper import *
from .basemodel import BaseModel


class LinearSVCModel(BaseModel):
    """
    LinearSVC classifier for the binary classification problem
    :param X_train: training data
    :param X_test: testing data
    :param y_train: training labels
    :param y_test: testing labels
    :return: LinearSVC classifier
    """

    def __init__(self, X_train, X_test, y_train, y_test, classes):
        super().__init__(
            "LinearSVC",
            svm.LinearSVC(
                C=1.0,
                class_weight=None,
                dual=True,
                fit_intercept=True,
                intercept_scaling=1,
                loss="squared_hinge",
                max_iter=5000,
                multi_class="ovr",
                penalty="l2",
                random_state=None,
                tol=0.0001,
                verbose=0,
            ),
            X_train,
            X_test,
            y_train,
            y_test,
            classes,
        )

def app_linearsvc(X_train, X_test, y_train, y_test, classes):
    """For Streamlit app"""
    linear_svc = LinearSVCModel(X_train, X_test, y_train, y_test, classes)
    linear_svc.run()
    linear_svc.save()
    st.pyplot(plt)
    st.write(linear_svc.results)
    st.markdown(
        get_download_link(f"output/models/classification/LinearSVC.joblib"),
        unsafe_allow_html=True,
    )

def main():
    """The main function"""
    X_train, X_test, y_train, y_test = load_gas_data("Data/data/expanded_data.csv")
    classes = {1: "Ethylene", 2: "Acetone", 3: "Ethylene + Acetone"}
    linear_svc = LinearSVCModel(X_train, X_test, y_train, y_test, classes)
    linear_svc.run()
    linear_svc.save()


if __name__ == "__main__":
    main()
