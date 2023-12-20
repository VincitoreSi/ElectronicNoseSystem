"""
file: linearsvc.py
author: @VincitoreSi
date: 2023-12-16
brief: LinearSVC classifier for the binary classification problem
"""

from dependencies.dependencies import *
from helper import plot_confusion_matrix, lead_and_prepare_data, load_gas_data


def linearsvc(X_train, X_test, y_train, y_test, classes):
    """
    LinearSVC classifier for the binary classification problem
    :param X_train: training data
    :param X_test: testing data
    :param y_train: training labels
    :param y_test: testing labels
    :return: LinearSVC classifier
    """
    svc_clf = svm.LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True, intercept_scaling=1, loss='squared_hinge', max_iter=5000, multi_class='ovr', penalty='l2', random_state=None, tol=0.0001, verbose=0)
    print("LinearSVC classifier")
    print("Training...")
    start = time.time()
    svc_clf.fit(X_train, y_train)
    end = time.time()
    print(f"Training time: {(end - start):.3f}s")
    print("Testing...")
    start = time.time()
    y_pred = svc_clf.predict(X_test)
    end = time.time()
    print(f"Testing time: {(end - start):.3f}s")
    print("Accuracy: ", svc_clf.score(X_test, y_test))
    print("Confusion matrix:")
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)
    cls = np.unique(y_train)
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=[classes[i] for i in cls], title='LinearSVC: Confusion matrix, without normalization')
    plt.savefig('output/images/linearsvc_cm.png')
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=[classes[i] for i in cls], normalize=True, title='LinearSVC: Normalized confusion matrix')
    plt.savefig('output/images/linearsvc_cm_normalized.png')
    # plt.show()
    return svc_clf

def main():
    """The main function"""
    X_train, X_test, y_train, y_test = load_gas_data('Data/data/expanded_data.csv')
    classes = {1: "Ethylene", 2: "Acetone", 3: "Ethylene + Acetone"}
    linearsvc(X_train, X_test, y_train, y_test, classes)

if __name__ == "__main__":
    main()