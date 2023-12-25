"""
file: app.py
author: @VincitoreSi
date: 2023-12-24
brief: Streamlit app for the project
"""

from sklearn.preprocessing import LabelEncoder
from joblib import dump
import base64
from dependencies.dependencies import *
from models.classification.adaboost import app_adaboost
from models.classification.baggingknn import app_baggingknn
from models.classification.decisiontree import app_decisiontree
from models.classification.extratree import app_extratree
from models.classification.knn import app_knn
from models.classification.linearsvc import app_linearsvc
from models.classification.logistic import app_logistic
from models.classification.naive_bayes import app_naive_bayes
from models.classification.random_forest import app_random_forest
from models.classification.voting import app_voting
from models.regression.svm import app_svm
from models.regression.linear import app_linear
from models.regression.ridge import app_ridge
from models.regression.elasticnet import app_elasticnet
from models.regression.sgd import app_sgd


from helper import *
from dimension_reduction import *

st.set_page_config(page_title="ElectronicNoseSystem", page_icon="🌫️", layout="wide")
st.title("Electronic Nose System for Gas Identification")

mode = st.sidebar.selectbox("Select Mode", ["Training", "Testing"])


if mode == "Training":
    st.header("Training")
    # File upload
    st.subheader("Upload CSV file")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        # Ask classes from user
        st.subheader("Enter Class Names")
        classes = {}
        cols = st.columns(len(data["GasType"].unique()))
        for i in range(len(data["GasType"].unique())):
            classes[i + 1] = cols[i].text_input(
                f"Enter class {i+1} name", value=f"Class {i+1}"
            )
        if st.button("Show Data"):
            st.subheader("Data")
            st.dataframe(data)

        # Preprocess data
        X_train, X_test, y_train, y_test = preprocess_data_clf(data)
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
        legend = []
        for i in range(len(data["GasType"].unique())):
            legend.append(classes[i + 1])
        if visualization_type == "PCA":
            pc = apply_pca(X, 2)
            pc_data = get_pca_data(pc, X)
            visualize_data(
                pc_data, y, "PCA_on_gas_sensor_binary_dataset", ["PC1", "PC2"], legend
            )
            st.pyplot(plt)
        elif visualization_type == "LDA":
            ld = apply_lda(X, y, 2)
            visualize_data(
                ld, y, "LDA_on_gas_sensor_binary_dataset", ["LD1", "LD2"], legend
            )
            st.pyplot(plt)
        elif visualization_type == "TSNE":
            tsne = apply_tsne(X, 2)
            visualize_data(
                tsne, y, "TSNE_on_gas_sensor_binary_dataset", ["TSNE1", "TSNE2"], legend
            )
            st.pyplot(plt)

        # Model selection
        model_type = st.selectbox(
            "Select Model Type",
            (
                "Classification",
                "Regression",
            ),
        )

        if model_type == "Classification":
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
                app_naive_bayes(X_train, X_test, y_train, y_test, classes)

            elif model == "Logistic Regression":
                app_logistic(X_train, X_test, y_train, y_test, classes)

            elif model == "Random Forest":
                app_random_forest(X_train, X_test, y_train, y_test, classes)

            elif model == "AdaBoost":
                app_adaboost(X_train, X_test, y_train, y_test, classes)

            elif model == "VotingClassifier":
                app_voting(X_train, X_test, y_train, y_test, classes)

            elif model == "KNN with Bagging":
                app_baggingknn(X_train, X_test, y_train, y_test, classes)

            elif model == "Decision Tree":
                app_decisiontree(X_train, X_test, y_train, y_test, classes)

            elif model == "Extra Trees":
                app_extratree(X_train, X_test, y_train, y_test, classes)

            elif model == "KNN":
                app_knn(X_train, X_test, y_train, y_test, classes)

            elif model == "Linear SVC":
                app_linearsvc(X_train, X_test, y_train, y_test, classes)

        elif model_type == "Regression":
            model = st.selectbox(
                "Select a model",
                (
                    "Linear Regression",
                    "Ridge Regression",
                    "Elastic Net",
                    "Support Vector Machine",
                    "SGD Regression",
                ),
            )
            # Train model
            if model == "Linear Regression":
                for i in range(len(data["GasType"].unique())):
                    app_linear(data, i + 1)

            elif model == "Ridge Regression":
                for i in range(len(data["GasType"].unique())):
                    app_ridge(data, i + 1)

            elif model == "Elastic Net":
                for i in range(len(data["GasType"].unique())):
                    app_elasticnet(data, i + 1)

            elif model == "Support Vector Machine":
                for i in range(len(data["GasType"].unique())):
                    app_svm(data, i + 1)

            elif model == "SGD Regression":
                for i in range(len(data["GasType"].unique())):
                    app_sgd(data, i + 1)

elif mode == "Testing":
    st.header("Testing")
    classes_num = st.number_input("Enter number of classes", min_value=1, max_value=10)
    st.subheader("Enter Class Names")
    classes = {}
    cols = st.columns(classes_num)
    for i in range(classes_num):
        classes[i + 1] = cols[i].text_input(
            f"Enter class {i+1} name", value=f"Class {i+1}"
        )
    # Predict
    st.subheader("Predict")
    # ask for sensor values
    sensor_num = st.number_input("Enter number of sensors", min_value=1, max_value=10)
    sensor_values = []
    # ask all of them in one line just like class names
    cols = st.columns(sensor_num)
    for i in range(sensor_num):
        sensor_values.append(cols[i].number_input(f"Enter sensor {i+1} value"))
    df = pd.DataFrame([sensor_values])
    df.columns = ["Sensor " + str(i + 1) for i in range(sensor_num)]
    print(df)
    # ask for model
    st.subheader("Select Classifier")
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
    if st.button("Predict Gas Type"):
        if model == "KNN":
            ans = load_model_and_predict(
                df,
                classes,
                "output/models/classification/KNeighborsClassifier.joblib",
            )
        elif model == "Naive Bayes":
            ans = load_model_and_predict(
                df, classes, "output/models/classification/GaussianNB.joblib"
            )

        elif model == "Logistic Regression":
            ans = load_model_and_predict(
                df,
                classes,
                "output/models/classification/LogisticRegression.joblib",
            )

        elif model == "Random Forest":
            ans = load_model_and_predict(
                df,
                classes,
                "output/models/classification/RandomForestClassifier.joblib",
            )

        elif model == "AdaBoost":
            ans = load_model_and_predict(
                df,
                classes,
                "output/models/classification/AdaBoostClassifier.joblib",
            )

        elif model == "VotingClassifier":
            ans = load_model_and_predict(
                df, classes, "output/models/classification/VotingClassifier.joblib"
            )

        elif model == "KNN with Bagging":
            ans = load_model_and_predict(
                df, classes, "output/models/classification/BaggingKNN.joblib"
            )

        elif model == "Decision Tree":
            ans = load_model_and_predict(
                df,
                classes,
                "output/models/classification/DecisionTreeClassifier.joblib",
            )

        elif model == "Extra Trees":
            ans = load_model_and_predict(
                df,
                classes,
                "output/models/classification/ExtraTreeClassifier.joblib",
            )

        elif model == "Linear SVC":
            ans = load_model_and_predict(
                df, classes, "output/models/classification/LinearSVC.joblib"
            )

    st.subheader("Select Regressor")
    ans, name = load_model_and_predict(
        df, classes, "output/models/classification/KNeighborsClassifier.joblib"
    )
    model = st.selectbox(
        "Select a model",
        (
            "Linear Regression",
            "Ridge Regression",
            "Elastic Net",
            "Support Vector Machine",
            "SGD Regression",
        ),
    )
    if st.button("Predict Gas Conc."):
        if model == "Linear Regression":
            if ans[0] < 3:
                load_model_and_predict_reg(df, f"output/models/regression/LinearRegression_{ans[0]}.joblib", name)
            else:
                name1 = name.split("+")[0]
                name2 = name.split("+")[1]
                load_model_and_predict_reg(df, f"output/models/regression/LinearRegression_{ans[0]}{0}.joblib", name1)
                load_model_and_predict_reg(df, f"output/models/regression/LinearRegression_{ans[0]}{1}.joblib", name2)
        
        elif model == "Ridge Regression":
            if ans[0] < 3:
                load_model_and_predict_reg(df, f"output/models/regression/BayesianRidge_{ans[0]}.joblib", name)
            else:
                name1 = name.split("+")[0]
                name2 = name.split("+")[1]
                load_model_and_predict_reg(df, f"output/models/regression/BayesianRidge_{ans[0]}{0}.joblib", name1)
                load_model_and_predict_reg(df, f"output/models/regression/BayesianRidge_{ans[0]}{1}.joblib", name2)
        
        elif model == "Elastic Net":
            if ans[0] < 3:
                load_model_and_predict_reg(df, f"output/models/regression/ElasticNet_{ans[0]}.joblib", name)
            else:
                name1 = name.split("+")[0]
                name2 = name.split("+")[1]
                load_model_and_predict_reg(df, f"output/models/regression/ElasticNet_{ans[0]}{0}.joblib", name1)
                load_model_and_predict_reg(df, f"output/models/regression/ElasticNet_{ans[0]}{1}.joblib", name2)
        
        elif model == "Support Vector Machine":
            if ans[0] < 3:
                load_model_and_predict_reg(df, f"output/models/regression/SVMRegression_{ans[0]}.joblib", name)
            else:
                name1 = name.split("+")[0]
                name2 = name.split("+")[1]
                load_model_and_predict_reg(df, f"output/models/regression/SVMRegression_{ans[0]}{0}.joblib", name1)
                load_model_and_predict_reg(df, f"output/models/regression/SVMRegression_{ans[0]}{1}.joblib", name2)

        elif model == "SGD Regression":
            if ans[0] < 3:
                load_model_and_predict_reg(df, f"output/models/regression/SGDRegressor_{ans[0]}.joblib", name)
            else:
                name1 = name.split("+")[0]
                name2 = name.split("+")[1]
                load_model_and_predict_reg(df, f"output/models/regression/SGDRegressor_{ans[0]}{0}.joblib", name1)
                load_model_and_predict_reg(df, f"output/models/regression/SGDRegressor_{ans[0]}{1}.joblib", name2)