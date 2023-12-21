"""
file: basemodel.py
author: @VincitoreSi
date: 2023-12-19
brief: Base model for the project
"""

from dependencies.dependencies import *
from helper import plot_confusion_matrix, lead_and_prepare_data, load_gas_data


class BaseModel:
    """
    Base model for the project
    """

    def __init__(self, model_name, model, X_train, X_test, y_train, y_test, classes):
        self.model_name = model_name
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.classes = classes
        self.results = {}

    def train(self):
        """
        Train the model
        :return: None
        """
        print(f"{self.model_name} classifier")
        print("Training...")
        start = time.time()
        self.model.fit(self.X_train, self.y_train)
        end = time.time()
        print(f"Training time: {(end - start):.3f}s")
        self.results["training_time"] = end - start

    def test(self):
        """
        Test the model
        :return: None
        """
        print("Testing...")
        start = time.time()
        y_pred = self.model.predict(self.X_test)
        end = time.time()
        print(f"Testing time: {(end - start):.3f}s")
        print("Accuracy: ", self.model.score(self.X_test, self.y_test))
        print("Confusion matrix:")
        cnf_matrix = confusion_matrix(self.y_test, y_pred)
        np.set_printoptions(precision=2)
        cls = np.unique(self.y_train)
        plt.figure()
        plot_confusion_matrix(
            cnf_matrix,
            classes=[self.classes[i] for i in cls],
            title=f"{self.model_name}: Confusion matrix, without normalization",
        )
        plt.savefig(f"output/images/{self.model_name}_cm.png")
        plt.figure()
        plot_confusion_matrix(
            cnf_matrix,
            classes=[self.classes[i] for i in cls],
            normalize=True,
            title=f"{self.model_name}: Normalized confusion matrix",
        )
        plt.savefig(f"output/images/{self.model_name}_cm_normalized.png")
        # plt.show()
        self.results["testing_time"] = end - start
        self.results["accuracy"] = self.model.score(self.X_test, self.y_test)
        self.results["confusion_matrix"] = cnf_matrix

    def run(self):
        """
        Run the model
        :return: None
        """
        self.train()
        self.test()
