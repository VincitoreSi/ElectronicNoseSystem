"""
file: extratree.py
author: @VincitoreSi
date: 2023-12-16
brief: Extra Tree classifier for the binary classification problem
"""

from dependencies.dependencies import *
from helper import plot_confusion_matrix, lead_and_prepare_data, load_gas_data

def extra_tree(X_train, X_test, y_train, y_test, max_depth, classes):
    """
    Extra Tree classifier for the binary classification problem
    :param X_train: training data
    :param X_test: testing data
    :param y_train: training labels
    :param y_test: testing labels
    :param max_depth: maximum depth
    :return: Extra Tree classifier
    """
    xt_clf = ExtraTreesClassifier(max_depth=max_depth)
    print("Extra Tree classifier")
    print("max_depth: ", max_depth)
    print("Training...")
    start = time.time()
    xt_clf.fit(X_train, y_train)
    end = time.time()
    print(f"Training time: {(end - start):.3f}s")
    print("Testing...")
    start = time.time()
    y_pred = xt_clf.predict(X_test)
    end = time.time()
    print(f"Testing time: {(end - start):.3f}s")
    print("Accuracy: ", xt_clf.score(X_test, y_test))
    print("Confusion matrix:")
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)
    cls = np.unique(y_train)
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=[classes[i] for i in cls], title='Extra Tree: Confusion matrix, without normalization')
    plt.savefig('output/images/extratree_cm.png')
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=[classes[i] for i in cls], normalize=True, title='Extra Tree: Normalized confusion matrix')
    plt.savefig('output/images/extratree_cm_normalized.png')
    # plt.show()
    return xt_clf

def main():
    """The main function"""
    X_train, X_test, y_train, y_test = load_gas_data('Data/data/expanded_data.csv')
    classes = {1: "Ethylene", 2: "Acetone", 3: "Ethylene + Acetone"}
    extra_tree(X_train, X_test, y_train, y_test, 10, classes)

if __name__ == "__main__":
    main()