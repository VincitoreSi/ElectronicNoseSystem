"""
file: main.py
author: @VincitoreSi
date: 2023-12-16
brief: Main file for the project
"""

from dependencies.dependencies import *

if __name__ == "__main__":
    # classification models
    os.system("python3 dimension_reduction.py > output/files/visualization.txt")
    os.system(
        "python3 -m models.classification.adaboost > output/files/classification/adaboost.txt"
    )
    os.system(
        "python3 -m models.classification.baggingknn > output/files/classification/baggingknn.txt"
    )
    os.system(
        "python3 -m models.classification.decisiontree > output/files/classification/decisiontree.txt"
    )
    os.system(
        "python3 -m models.classification.extratree > output/files/classification/extratree.txt"
    )
    os.system("python3 -m models.classification.knn > output/files/classification/knn.txt")
    os.system(
        "python3 -m models.classification.linearsvc > output/files/classification/linearsvc.txt"
    )
    os.system(
        "python3 -m models.classification.logistic > output/files/classification/logistic.txt"
    )
    os.system(
        "python3 -m models.classification.naive_bayes > output/files/classification/naive_bayes.txt"
    )
    os.system(
        "python3 -m models.classification.random_forest > output/files/classification/random_forest.txt"
    )
    os.system(
        "python3 -m models.classification.voting > output/files/classification/voting.txt"
    )
    # regression models
    os.system("python3 -m models.regression.linear > output/files/regression/linear.txt")
    os.system("python3 -m models.regression.svm > output/files/regression/svm.txt")
    os.system("python3 -m models.regression.sgd > output/files/regression/sgd.txt")
    os.system("python3 -m models.regression.ridge > output/files/regression/ridge.txt")
    os.system("python3 -m models.regression.elasticnet > output/files/regression/elasticnet.txt")

