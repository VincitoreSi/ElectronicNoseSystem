"""
file: extratree.py
author: @VincitoreSi
date: 2023-12-16
brief: Extra Tree classifier for the binary classification problem
"""

from dependencies.dependencies import *
from helper import *
from .basemodel import BaseModel


class ExtraTree(BaseModel):
    """
    Extra Tree classifier for the binary classification problem
    :param X_train: training data
    :param X_test: testing data
    :param y_train: training labels
    :param y_test: testing labels
    :param max_depth: maximum depth
    :return: Extra Tree classifier
    """

    def __init__(self, X_train, X_test, y_train, y_test, max_depth, classes):
        super().__init__(
            "ExtraTreeClassifier",
            ExtraTreesClassifier(max_depth=max_depth),
            X_train,
            X_test,
            y_train,
            y_test,
            classes,
        )

def app_extratree(X_train, X_test, y_train, y_test, classes):
    """For Streamlit app"""
    extra_tree = ExtraTree(X_train, X_test, y_train, y_test, 10, classes)
    extra_tree.run()
    extra_tree.save()
    st.pyplot(plt)
    st.write(extra_tree.results)
    st.markdown(
        get_download_link(f"output/models/classification/ExtraTreeClassifier.joblib"),
        unsafe_allow_html=True,
    )

def main():
    """The main function"""
    X_train, X_test, y_train, y_test = load_gas_data("Data/data/expanded_data.csv")
    classes = {1: "Ethylene", 2: "Acetone", 3: "Ethylene + Acetone"}
    extra_tree = ExtraTree(X_train, X_test, y_train, y_test, 10, classes)
    extra_tree.run()
    extra_tree.save()


if __name__ == "__main__":
    main()
