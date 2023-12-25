"""
file: svm.py
author: @VincitoreSi
date: 2023-12-21
brief: SVM model for predicting the concentration of the gases
"""

from dependencies.dependencies import *
from helper import *
from .basemodel import BaseModel


class SVMRegressionModel(BaseModel):
    """
    SVM model for predicting the concentration of the gases
    :param X_train: training data
    :param X_test: testing data
    :param y_train: training labels
    :param y_test: testing labels
    :return: SVM model
    """

    def __init__(self, X_train, X_test, y_train, y_test):
        super().__init__(
            "SVMRegression",
            SVR(),
            X_train,
            X_test,
            y_train,
            y_test,
        )


def app_svm(data, cls):
    """For Streamlit app"""
    X_train, X_test, y_train, y_test = preprocess_data_reg(data, cls)
    shape = 1 if len(y_train.shape) == 1 else y_train.shape[1]
    print(f"Shape: {shape}")
    if shape == 1:
        svm_regression = SVMRegressionModel(X_train, X_test, y_train, y_test)
        svm_regression.run()
        svm_regression.save(cls)
        results_df = pd.DataFrame([svm_regression.results])
        results_df = results_df.reset_index(drop=True)
        print(results_df)
        st.table(results_df)
        st.markdown(
            get_download_link(f"output/models/regression/SVMRegression_{cls}.joblib"),
            unsafe_allow_html=True,
        )
    else:
        for i in range(shape):
            svm_regression = SVMRegressionModel(
                X_train, X_test, y_train.iloc[:, i], y_test.iloc[:, i]
            )
            svm_regression.run()
            svm_regression.save(cls, i)
            results_df = pd.DataFrame([svm_regression.results])
            results_df = results_df.reset_index(drop=True)
            st.table(results_df)
            st.markdown(
                get_download_link(
                    f"output/models/regression/SVMRegression_{cls}{i}.joblib"
                ),
                unsafe_allow_html=True,
            )


def main(cls):
    """The main function"""
    X_train, X_test, y_train, y_test = load_gas_data_for_regression(
        "Data/data/expanded_data.csv", cls
    )
    shape = 1 if len(y_train.shape) == 1 else y_train.shape[1]
    if shape == 1:
        svm_regression = SVMRegressionModel(X_train, X_test, y_train, y_test)
        svm_regression.run()
        svm_regression.save(cls)
    else:
        for i in range(shape):
            svm_regression = SVMRegressionModel(
                X_train, X_test, y_train.iloc[:, i], y_test.iloc[:, i]
            )
            svm_regression.run()
            svm_regression.save(cls)


if __name__ == "__main__":
    for i in range(1, 4):
        main(i)
