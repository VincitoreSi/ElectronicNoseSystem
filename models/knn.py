"""
file: knn.py
author: @VincitoreSi
date: 2023-12-16
brief: KNN classifier for the binary classification problem
"""

from dependencies.dependencies import *
from helper import plot_confusion_matrix, lead_and_prepare_data, load_gas_data

def knn(X_train, X_test, y_train, y_test, n_neighbors, classes):
    """
    KNN classifier for the binary classification problem
    :param X_train: training data
    :param X_test: testing data
    :param y_train: training labels
    :param y_test: testing labels
    :param n_neighbors: number of neighbors
    :return: KNN classifier
    """
    knn_clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    print("KNN classifier")
    print("n_neighbors: ", n_neighbors)
    print("Training...")
    start = time.time()
    knn_clf.fit(X_train, y_train)
    end = time.time()
    print(f"Training time: {(end - start):.3f}s")
    print("Testing...")
    start = time.time()
    y_pred = knn_clf.predict(X_test)
    end = time.time()
    print(f"Testing time: {(end - start):.3f}s")
    print("Accuracy: ", knn_clf.score(X_test, y_test))
    print("Confusion matrix:")
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)
    cls = np.unique(y_train)
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=[classes[i] for i in cls], title='KNN: Confusion matrix, without normalization')
    plt.savefig('output/images/knn_cm.png')
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=[classes[i] for i in cls], normalize=True, title='KNN: Normalized confusion matrix')
    plt.savefig('output/images/knn_cm_normalized.png')
    # plt.show()
    return knn_clf

def main():
    """The main function"""
    X_train, X_test, y_train, y_test = load_gas_data('Data/data/expanded_data.csv')
    classes = {1: "Ethylene", 2: "Acetone", 3: "Ethylene + Acetone"}
    knn(X_train, X_test, y_train, y_test, 14, classes)

if __name__ == "__main__":
    main()