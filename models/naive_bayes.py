"""
file: naive_bayes.py
author: @VincitoreSi
date: 2023-12-16
brief: Naive Bayes classifier for the binary classification problem
"""

from dependencies.dependencies import *
from helper import plot_confusion_matrix, lead_and_prepare_data, load_gas_data

def naive_bayes(X_train, X_test, y_train, y_test, classes):
    """
    Naive Bayes classifier for the binary classification problem
    :param X_train: training data
    :param X_test: testing data
    :param y_train: training labels
    :param y_test: testing labels
    :return: Naive Bayes classifier
    """
    print("Naive Bayes classifier")
    print("Training...")
    start = time.time()
    clf = GaussianNB()
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
    plot_confusion_matrix(cnf_matrix, classes=[classes[i] for i in cls], title='Naive Bayes: Confusion matrix, without normalization')
    plt.savefig('output/images/naivebayes_cm.png')
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=[classes[i] for i in cls], normalize=True, title='Naive Bayes: Normalized confusion matrix')
    plt.savefig('output/images/naivebayes_cm_normalized.png')
    # plt.show()
    return clf

def main():
    """The main function"""
    X_train, X_test, y_train, y_test = load_gas_data('Data/data/expanded_data.csv')
    classes = {1: "Ethylene", 2: "Acetone", 3: "Ethylene + Acetone"}
    naive_bayes(X_train, X_test, y_train, y_test, classes)

if __name__ == "__main__":
    main()