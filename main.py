"""
file: main.py
author: @VincitoreSi
date: 2023-12-16
brief: Main file for the project
"""

from dependencies.dependencies import *

if __name__ == "__main__":
    # execute commands in the terminal
    os.system("python3 dimension_reduction.py > output/files/visualization.txt")
    os.system("python3 -m models.classification.adaboost > output/files/adaboost.txt")
    os.system("python3 -m models.classification.baggingknn > output/files/baggingknn.txt")
    os.system("python3 -m models.classification.decisiontree > output/files/decisiontree.txt")
    os.system("python3 -m models.classification.extratree > output/files/extratree.txt")
    os.system("python3 -m models.classification.knn > output/files/knn.txt")
    os.system("python3 -m models.classification.linearsvc > output/files/linearsvc.txt")
    os.system("python3 -m models.classification.logistic > output/files/logistic.txt")
    os.system("python3 -m models.classification.naive_bayes > output/files/naive_bayes.txt")
    os.system("python3 -m models.classification.random_forest > output/files/random_forest.txt")
    os.system("python3 -m models.classification.voting > output/files/voting.txt")
