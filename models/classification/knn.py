"""
file: knn.py
author: @VincitoreSi
date: 2023-12-16
brief: KNN classifier for the binary classification problem
"""

from dependencies.dependencies import *
from helper import *
from .basemodel import BaseModel


class KNN(BaseModel):
    """
    KNN classifier for the binary classification problem
    :param X_train: training data
    :param X_test: testing data
    :param y_train: training labels
    :param y_test: testing labels
    :param n_neighbors: number of neighbors
    :return: KNN classifier
    """

    def __init__(self, X_train, X_test, y_train, y_test, n_neighbors, classes):
        super().__init__(
            "KNeighborsClassifier",
            KNeighborsClassifier(n_neighbors=n_neighbors),
            X_train,
            X_test,
            y_train,
            y_test,
            classes,
        )

def app_knn(X_train, X_test, y_train, y_test, classes):
    """For Streamlit app"""
    knn = KNN(X_train, X_test, y_train, y_test, 14, classes)
    knn.run()
    knn.save()
    st.pyplot(plt)
    st.write(knn.results)
    st.markdown(
        get_download_link(f"output/models/classification/KNeighborsClassifier.joblib"),
        unsafe_allow_html=True,
    )

def main():
    """The main function"""
    X_train, X_test, y_train, y_test = load_gas_data("Data/data/data_update.csv")
    # classes = {1: "Ethylene", 2: "Acetone", 3: "Ethylene + Acetone"}
    classes = {1: "Ethylene", 2: "Acetone", 3: "A", 4: "B", 5: "C", 6: "D"}
    knn = KNN(X_train, X_test, y_train, y_test, 14, classes)
    knn.run()
    knn.save()

if __name__ == "__main__":
    main()
