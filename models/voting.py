"""
file: voting.py
author: @VincitoreSi
date: 2023-12-16
brief: Voting classifier for the binary classification problem
"""

from dependencies.dependencies import *
from helper import plot_confusion_matrix, lead_and_prepare_data, load_gas_data

def majority_voting(X_train, X_test, y_train, y_test, classes):
    """
    Majority voting classifier for the binary classification problem
    :param X_train: training data
    :param X_test: testing data
    :param y_train: training labels
    :param y_test: testing labels
    :return: Majority voting classifier
    """
    clf1 = LogisticRegression()
    clf2= DecisionTreeClassifier()
    clf3= LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
         intercept_scaling=1, loss='squared_hinge', max_iter=5000,
         multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
         verbose=0)
    clf4= KNeighborsClassifier(n_neighbors=30)
    eclf = VotingClassifier(estimators=[('lr', clf1), ('dt', clf2), ('svm', clf3),('knn',clf4)], voting='hard')
    eclf = eclf.fit(X_train,y_train)
    y_pred =eclf.predict(X_test)
    print("Majority voting classifier")
    print("Training...")
    start = time.time()
    eclf.fit(X_train, y_train)
    end = time.time()
    print(f"Training time: {(end - start):.3f}s")
    print("Testing...")
    start = time.time()
    y_pred = eclf.predict(X_test)
    end = time.time()
    print(f"Testing time: {(end - start):.3f}s")
    print("Accuracy: ", eclf.score(X_test, y_test))
    print("Confusion matrix:")
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)
    cls = np.unique(y_train)
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=[classes[i] for i in cls], title='Majority voting: Confusion matrix, without normalization')
    plt.savefig('output/images/majority_voting_cm.png')
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=[classes[i] for i in cls], normalize=True, title='Majority voting: Normalized confusion matrix')
    plt.savefig('output/images/majority_voting_cm_normalized.png')
    # plt.show()
    return eclf

def main():
    """The main function"""
    X_train, X_test, y_train, y_test = load_gas_data('Data/data/expanded_data.csv')
    classes = {1: "Ethylene", 2: "Acetone", 3: "Ethylene + Acetone"}
    majority_voting(X_train, X_test, y_train, y_test, classes)

if __name__ == "__main__":
    main()