"""
file: random_forest.py
author: @VincitoreSi
date: 2023-12-16
brief: Random Forest classifier for the binary classification problem
"""

from dependencies.dependencies import *
from helper import *
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

def app_random_forest(X_train, X_test, y_train, y_test, classes):
    """For Streamlit app"""
    random_forest = RandomForest(X_train, X_test, y_train, y_test, 100, classes)
    random_forest.run()
    random_forest.save()
    st.pyplot(plt)
    st.write(random_forest.results)
    st.markdown(
        get_download_link(f"output/models/classification/RandomForestClassifier.joblib"),
        unsafe_allow_html=True,
    )

def main():
    """The main function"""
    X_train, X_test, y_train, y_test = load_gas_data("Data/data/expanded_data.csv")
    classes = {1: "Ethylene", 2: "Acetone", 3: "Ethylene + Acetone"}
    random_forest = RandomForest(X_train, X_test, y_train, y_test, 100, classes)
    random_forest.run()
    random_forest.save()

if __name__ == "__main__":
    main()
