"""
file: basemodel.py
author: @VincitoreSi
date: 2023-12-21
brief: Base model for the regression problem
"""

from dependencies.dependencies import *
from helper import plot_confusion_matrix, lead_and_prepare_data, load_gas_data


class BaseModel:
    """
    Base model for the regression problem
    """

    def __init__(self, name, model, X_train, X_test, y_train, y_test):
        """
        Constructor
        :param name: name of the model
        :param model: model
        :param X_train: training data
        :param X_test: testing data
        :param y_train: training labels
        :param y_test: testing labels
        """
        self.name = name
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train.values.ravel()
        self.y_test = y_test.values.ravel()
        self.results = {}

    def train(self):
        """
        Train the model
        """
        start = time.time()
        self.model.fit(self.X_train, self.y_train)
        end = time.time()
        self.results["train_time"] = end - start

    def test(self):
        """
        Test the model
        """
        start = time.time()
        y_pred = self.model.predict(self.X_test)
        end = time.time()
        self.results["test_time"] = end - start
        print(f"Model: {self.name}")
        print(f"Mean Absolute Error: {mean_absolute_error(self.y_test, y_pred):.3f}")
        print(f"Mean Squared Error: {mean_squared_error(self.y_test, y_pred):.3f}")
        print(f"Median Absolute Error: {median_absolute_error(self.y_test, y_pred):.3f}")
        print(f"R2 Score: {r2_score(self.y_test, y_pred):.3f}")
        print(f"Explained Variance Score: {explained_variance_score(self.y_test, y_pred):.3f}")
        print(f"Max Error: {max_error(self.y_test, y_pred):.3f}")
        print(f"Mean Squared Log Error: {mean_squared_log_error(self.y_test, y_pred):.3f}")
        print(f"Training Time: {self.results['train_time']:.3f}")
        print(f"Test Time: {self.results['test_time']:.3f}")
        self.results["MAE"] = mean_absolute_error(self.y_test, y_pred)
        self.results["MSE"] = mean_squared_error(self.y_test, y_pred)
        self.results["MedAE"] = median_absolute_error(self.y_test, y_pred)
        self.results["R2"] = r2_score(self.y_test, y_pred)
        self.results["EVS"] = explained_variance_score(self.y_test, y_pred)
        self.results["ME"] = max_error(self.y_test, y_pred)
        self.results["MSLE"] = mean_squared_log_error(self.y_test, y_pred)

    def predict(self, X):
        """
        Predict the concentration of the gases
        :param X: data
        :return: predicted values
        """
        return self.model.predict(X)