"""
file: app.py
author: @VincitoreSi
date: 2023-12-24
brief: Streamlit app for the project
"""

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from joblib import dump
import base64
from dependencies.dependencies import *
from models.classification.adaboost import AdaBoost
from models.classification.baggingknn import BaggingKNN
from models.classification.decisiontree import DecisionTree
from models.classification.extratree import ExtraTree
from models.classification.knn import KNN
from models.classification.linearsvc import LinearSVCModel
from models.classification.logistic import LogisticModel
from models.classification.naive_bayes import NaiveBayes
from models.classification.random_forest import RandomForest
from models.classification.voting import VotingModel
from models.regression.svm import SVMRegressionModel
from models.regression.linear import LinearRegressionModel
from models.regression.ridge import RidgeRegressionModel
from models.regression.elasticnet import ElasticNetRegressionModel


from helper import *
from dimension_reduction import *

st.set_page_config(page_title="ElectronicNoseSystem", page_icon="🌫️", layout="wide")
st.title("Electronic Nose System for Gas Identification")

mode = st.sidebar.selectbox("Select Mode", ["Training", "Testing"])

if mode == "Training":
    st.header("Training")
    # File upload

elif mode == "Testing":
    st.header("Testing")

    # Predict
    st.subheader("Predict")
    if st.button("Predict"):
        st.write("Predicted")

    # Evaluation
    st.subheader("Evaluation")
    if st.button("Evaluate"):
        st.write("Evaluated")

    # Visualization
    st.subheader("Visualization")
    if st.button("Visualize"):
        st.write("Visualized")