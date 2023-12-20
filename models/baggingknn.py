"""
file: baggingknn.py
author: @VincitoreSi
date: 2023-12-16
brief: Bagging classifier with KNN
"""

from dependencies.dependencies import *
from helper import plot_confusion_matrix, lead_and_prepare_data, load_gas_data

def bagging_with_knn(X_train, X_test, y_train, y_test, n_estimators, max_samples, max_features, classes):
    """
    This function implements bagging with KNN classifier
    :param X_train: training data
    :param X_test: testing data
    :param y_train: training labels
    :param y_test: testing labels
    :param n_estimators: number of estimators
    :param max_samples: maximum samples
    :param max_features: maximum features
    :return: Bagging classifier with KNN
    """
    bagging_clf = BaggingClassifier(KNeighborsClassifier(n_neighbors=14), n_estimators=n_estimators, max_samples=max_samples, max_features=max_features)
    print("Bagging classifier with KNN")
    print("n_estimators: ", n_estimators)
    print("max_samples: ", max_samples)
    print("max_features: ", max_features)
    print("Training...")
    start = time.time()
    bagging_clf.fit(X_train, y_train)
    end = time.time()
    print(f"Training time: {(end - start):.3f}s")
    print("Testing...")
    start = time.time()
    y_pred = bagging_clf.predict(X_test)
    end = time.time()
    print(f"Testing time: {(end - start):.3f}s")
    print("Accuracy: ", bagging_clf.score(X_test, y_test))
    print("Confusion matrix:")
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)
    cls = np.unique(y_train)
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=[classes[i] for i in cls], title='KNN + Bagging: Confusion matrix, without normalization')
    plt.savefig('output/images/baggingknn_cm.png')
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=[classes[i] for i in cls], normalize=True, title='KNN + Bagging: Normalized confusion matrix')
    plt.savefig('output/images/baggingknn_cm_normalized.png')
    # plt.show()
    return bagging_clf


def main():
    """The main function"""
    X_train, X_test, y_train, y_test = load_gas_data('Data/data/expanded_data.csv')
    classes = {1: "Ethylene", 2: "Acetone", 3: "Ethylene + Acetone"}
    bagging_with_knn(X_train, X_test, y_train, y_test, 100, 0.5, 0.5, classes)

if __name__ == "__main__":
    main()