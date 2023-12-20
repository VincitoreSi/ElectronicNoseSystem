import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from joblib import dump
import base64
from dependencies.dependencies import *
from models.adaboost import AdaBoost
from models.baggingknn import BaggingKNN
from models.decisiontree import DecisionTree
from models.extratree import ExtraTree
from models.knn import KNN
from models.linearsvc import LinearSVCModel
from models.logistic import LogisticModel
from models.naive_bayes import NaiveBayes
from models.random_forest import RandomForest
from models.voting import VotingModel


from helper import *
from dimension_reduction import *


# Function to download model
def get_download_link(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{file_path}">Download Trained Model</a>'
    return href


# Title
st.title("Gas Classification")


# File upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(data)

    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(data)

    visualization_type = st.selectbox(
        "Select a visualization type",
        (
            "PCA",
            "LDA",
            "TSNE",
        ),
    )
    X, y = np.array(X_train), np.array(y_train)

    # show plot based on visualization type
    if visualization_type == "PCA":
        pc = apply_pca(X, 2)
        pc_data = get_pca_data(pc, X)
        visualize_data(pc_data, y, "PCA_on_gas_sensor_binary_dataset", ["PC1", "PC2"])
        st.pyplot(plt)
    elif visualization_type == "LDA":
        ld = apply_lda(X, y, 2)
        visualize_data(ld, y, "LDA_on_gas_sensor_binary_dataset", ["LD1", "LD2"])
        st.pyplot(plt)
    elif visualization_type == "TSNE":
        tsne = apply_tsne(X, 2)
        visualize_data(tsne, y, "TSNE_on_gas_sensor_binary_dataset", ["TSNE1", "TSNE2"])
        st.pyplot(plt)

    # Ask classes from user
    classes = {}
    for i in range(1, len(data["GasType"].unique()) + 1):
        classes[i] = st.text_input(f"Enter class {i} name", value=f"Class {i}")

    # Model selection
    model = st.selectbox(
        "Select a model",
        (
            "Naive Bayes",
            "Logistic Regression",
            "Random Forest",
            "AdaBoost",
            "VotingClassifier",
            "KNN with Bagging",
            "Decision Tree",
            "Extra Trees",
            "KNN",
            "Linear SVC",
        ),
    )

    # Train model
    if model == "Naive Bayes":
        naive_bayes = NaiveBayes(X_train, X_test, y_train, y_test, classes)
        naive_bayes.train()
        naive_bayes.test()
        dump(naive_bayes.model, "output/models/naive_bayes.joblib")
        st.markdown(
            get_download_link("output/models/naive_bayes.joblib"),
            unsafe_allow_html=True,
        )
        st.pyplot(plt)
        st.write(f"Training time: {naive_bayes.results['training_time']: .3f}s")
        st.write(f"Testing time: {naive_bayes.results['testing_time']: .3f}s")
        st.write(f"Accuracy: {naive_bayes.results['accuracy']: .3f}")

    elif model == "Logistic Regression":
        logistic_regression = LogisticModel(X_train, X_test, y_train, y_test, classes)
        logistic_regression.train()
        logistic_regression.test()
        dump(logistic_regression.model, "output/models/logistic_regression.joblib")
        st.markdown(
            get_download_link("output/models/logistic_regression.joblib"),
            unsafe_allow_html=True,
        )
        st.pyplot(plt)
        st.write(f"Training time: {logistic_regression.results['training_time']: .3f}s")
        st.write(f"Testing time: {logistic_regression.results['testing_time']: .3f}s")
        st.write(f"Accuracy: {logistic_regression.results['accuracy']: .3f}")

    elif model == "Random Forest":
        random_forest = RandomForest(X_train, X_test, y_train, y_test, 100, classes)
        random_forest.train()
        random_forest.test()
        dump(random_forest.model, "output/models/random_forest.joblib")
        st.markdown(
            get_download_link("output/models/random_forest.joblib"),
            unsafe_allow_html=True,
        )
        st.pyplot(plt)
        st.write(f"Training time: {random_forest.results['training_time']: .3f}s")
        st.write(f"Testing time: {random_forest.results['testing_time']: .3f}s")
        st.write(f"Accuracy: {random_forest.results['accuracy']: .3f}")

    elif model == "AdaBoost":
        adaboost = AdaBoost(X_train, X_test, y_train, y_test, 100, 1, classes)
        adaboost.train()
        adaboost.test()
        dump(adaboost.model, "output/models/adaboost.joblib")
        st.markdown(
            get_download_link("output/models/adaboost.joblib"), unsafe_allow_html=True
        )
        st.pyplot(plt)
        st.write(f"Training time: {adaboost.results['training_time']: .3f}s")
        st.write(f"Testing time: {adaboost.results['testing_time']: .3f}s")
        st.write(f"Accuracy: {adaboost.results['accuracy']: .3f}")

    elif model == "VotingClassifier":
        voting = VotingModel(X_train, X_test, y_train, y_test, classes)
        voting.train()
        voting.test()
        dump(voting.model, "output/models/voting.joblib")
        st.markdown(
            get_download_link("output/models/voting.joblib"), unsafe_allow_html=True
        )
        st.pyplot(plt)
        st.write(f"Training time: {voting.results['training_time']: .3f}s")
        st.write(f"Testing time: {voting.results['testing_time']: .3f}s")
        st.write(f"Accuracy: {voting.results['accuracy']: .3f}")

    elif model == "KNN with Bagging":
        baggingknn = BaggingKNN(X_train, X_test, y_train, y_test, classes, 100, 0.5, 0.5)
        baggingknn.train()
        baggingknn.test()
        dump(baggingknn.model, "output/models/baggingknn.joblib")
        st.markdown(
            get_download_link("output/models/baggingknn.joblib"), unsafe_allow_html=True
        )
        st.pyplot(plt)
        st.write(f"Training time: {baggingknn.results['training_time']: .3f}s")
        st.write(f"Testing time: {baggingknn.results['testing_time']: .3f}s")
        st.write(f"Accuracy: {baggingknn.results['accuracy']: .3f}")

    elif model == "Decision Tree":
        decisiontree = DecisionTree(X_train, X_test, y_train, y_test, 10, classes)
        decisiontree.train()
        decisiontree.test()
        dump(decisiontree.model, "output/models/decisiontree.joblib")
        st.markdown(
            get_download_link("output/models/decisiontree.joblib"),
            unsafe_allow_html=True,
        )
        st.pyplot(plt)
        st.write(f"Training time: {decisiontree.results['training_time']: .3f}s")
        st.write(f"Testing time: {decisiontree.results['testing_time']: .3f}s")
        st.write(f"Accuracy: {decisiontree.results['accuracy']: .3f}")

    elif model == "Extra Trees":
        extratree = ExtraTree(X_train, X_test, y_train, y_test, classes)
        extratree.train()
        extratree.test()
        dump(extratree.model, "output/models/extratree.joblib")
        st.markdown(
            get_download_link("output/models/extratree.joblib"), unsafe_allow_html=True
        )
        st.pyplot(plt)
        st.write(f"Training time: {extratree.results['training_time']: .3f}s")
        st.write(f"Testing time: {extratree.results['testing_time']: .3f}s")
        st.write(f"Accuracy: {extratree.results['accuracy']: .3f}")

    elif model == "KNN":
        knn = KNN(X_train, X_test, y_train, y_test, 14, classes)
        knn.train()
        knn.test()
        dump(knn.model, "output/models/knn.joblib")
        st.markdown(
            get_download_link("output/models/knn.joblib"), unsafe_allow_html=True
        )
        st.pyplot(plt)
        st.write(f"Training time: {knn.results['training_time']: .3f}s")
        st.write(f"Testing time: {knn.results['testing_time']: .3f}s")
        st.write(f"Accuracy: {knn.results['accuracy']: .3f}")

    elif model == "Linear SVC":
        linear_svc = LinearSVCModel(X_train, X_test, y_train, y_test, classes)
        linear_svc.train()
        linear_svc.test()
        dump(linear_svc.model, "output/models/linear_svc.joblib")
        st.markdown(
            get_download_link("output/models/linear_svc.joblib"), unsafe_allow_html=True
        )
        st.pyplot(plt)
        st.write(f"Training time: {linear_svc.results['training_time']: .3f}s")
        st.write(f"Testing time: {linear_svc.results['testing_time']: .3f}s")
        st.write(f"Accuracy: {linear_svc.results['accuracy']: .3f}")

    else:
        st.write("Model not implemented yet")
