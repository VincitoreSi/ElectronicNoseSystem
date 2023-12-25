"""
file: voting.py
author: @VincitoreSi
date: 2023-12-16
brief: Voting classifier for the binary classification problem
"""

from dependencies.dependencies import *
from helper import *
from .basemodel import BaseModel


class VotingModel(BaseModel):
    """
    Voting classifier for the binary classification problem
    :param X_train: training data
    :param X_test: testing data
    :param y_train: training labels
    :param y_test: testing labels
    :return: Voting classifier
    """

    def __init__(self, X_train, X_test, y_train, y_test, classes):
        clf1 = LogisticRegression()
        clf2 = DecisionTreeClassifier()
        clf3 = LinearSVC(
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
        )
        clf4 = KNeighborsClassifier(n_neighbors=30)
        eclf = VotingClassifier(
            estimators=[("lr", clf1), ("dt", clf2), ("svm", clf3), ("knn", clf4)],
            voting="hard",
        )
        super().__init__(
            "VotingClassifier", eclf, X_train, X_test, y_train, y_test, classes
        )

def app_voting(X_train, X_test, y_train, y_test, classes):
    """For Streamlit app"""
    voting = VotingModel(X_train, X_test, y_train, y_test, classes)
    voting.run()
    voting.save()
    st.pyplot(plt)
    st.write(voting.results)
    st.markdown(
        get_download_link(f"output/models/classification/VotingClassifier.joblib"),
        unsafe_allow_html=True,
    )

def main():
    """The main function"""
    X_train, X_test, y_train, y_test = load_gas_data("Data/data/expanded_data.csv")
    classes = {1: "Ethylene", 2: "Acetone", 3: "Ethylene + Acetone"}
    voting = VotingModel(X_train, X_test, y_train, y_test, classes)
    voting.run()
    voting.save()

if __name__ == "__main__":
    main()
