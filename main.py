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
    os.system("python3 -m models.adaboost > output/files/adaboost.txt")
    os.system("python3 -m models.baggingknn > output/files/baggingknn.txt")
    os.system("python3 -m models.decisiontree > output/files/decisiontree.txt")
    os.system("python3 -m models.extratree > output/files/extratree.txt")
    os.system("python3 -m models.knn > output/files/knn.txt")
    os.system("python3 -m models.linearsvc > output/files/linearsvc.txt")
    os.system("python3 -m models.logistic > output/files/logistic.txt")
    os.system("python3 -m models.naive_bayes > output/files/naive_bayes.txt")
    os.system("python3 -m models.random_forest > output/files/random_forest.txt")
    os.system("python3 -m models.voting > output/files/voting.txt")