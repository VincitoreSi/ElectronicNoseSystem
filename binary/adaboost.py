"""
file: adaboost.py
author: @VincitoreSi
date: 2023-12-16
brief: AdaBoost classifier for the binary classification problem
"""

from dependencies.dependencies import *
from helper import plot_confusion_matrix, lead_and_prepare_data

def adaboost(X_train, X_test, y_train, y_test, n_estimators, learning_rate):
    """
    AdaBoost classifier for the binary classification problem
    :param X_train: training data
    :param X_test: testing data
    :param y_train: training labels
    :param y_test: testing labels
    :param n_estimators: number of estimators
    :param learning_rate: learning rate
    :param max_depth: maximum depth
    :return: AdaBoost classifier
    """
    print("AdaBoost classifier")
    print("n_estimators: ", n_estimators)
    print("learning_rate: ", learning_rate)
    print("Training...")
    start = time.time()
    clf = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate)
    clf.fit(X_train, y_train)
    end = time.time()
    print("Training time: ", end - start)
    print("Testing...")
    start = time.time()
    y_pred = clf.predict(X_test)
    end = time.time()
    print("Testing time: ", end - start)
    print("Accuracy: ", clf.score(X_test, y_test))
    print("Confusion matrix:")
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=["0", "1"], title='Confusion matrix, without normalization')
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=["0", "1"], normalize=True, title='Normalized confusion matrix')
    plt.show()
    return clf

def main():
    """
    Main function
    """
    X_train, X_test, y_train, y_test = lead_and_prepare_data()
    clf = adaboost(X_train, X_test, y_train, y_test, 100, 1)
    
if __name__ == "__main__":
    main()