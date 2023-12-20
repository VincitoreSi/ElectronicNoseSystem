"""
file: adaboost.py
author: @VincitoreSi
date: 2023-12-16
brief: AdaBoost classifier for the binary classification problem
"""

from dependencies.dependencies import *
from helper import plot_confusion_matrix, lead_and_prepare_data, load_gas_data

def adaboost(X_train, X_test, y_train, y_test, n_estimators, learning_rate, classes):
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
    print(f"Training time: {(end - start):.3f}s")
    print("Testing...")
    start = time.time()
    y_pred = clf.predict(X_test)
    end = time.time()
    print(f"Testing time: {(end - start):.3f}s")
    print("Accuracy: ", clf.score(X_test, y_test))
    print("Confusion matrix:")
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)
    cls = np.unique(y_train)
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=[classes[i] for i in cls], title='AdaBoostClassifier: Confusion matrix, without normalization')
    plt.savefig('output/images/adaboost_cm.png')
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=[classes[i] for i in cls], normalize=True, title='AdaBoostClassifier: Normalized confusion matrix')
    plt.savefig('output/images/adaboost_cm_normalized.png')
    # plt.show()
    return clf

def main():
    """
    Main function
    """
    X_train, X_test, y_train, y_test = load_gas_data('Data/data/expanded_data.csv')
    classes = {1: "Ethylene", 2: "Acetone", 3: "Ethylene + Acetone"}
    clf = adaboost(X_train, X_test, y_train, y_test, 100, 1, classes)
    
if __name__ == "__main__":
    main()