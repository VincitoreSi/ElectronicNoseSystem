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

with open("style.css") as css:
    st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)


mode = option_menu(
    menu_title="Main Menu",
    menu_icon="cast",
    options=["Home", "Training", "Testing", "Contact Us"],
    icons=["🏠", "📚", "🧪", "📞"],
    default_index=0,
    orientation="horizontal",
)   

if mode == "Home":
    col1, col2 = st.columns([1, 3], gap="large")

    col1.markdown(
        "# Electronic Nose System for Gas Identification and Concentration Estimation \n \n"
    )
    
    col1.image("images/raspberry-pi-logo.png", use_column_width=True)
    col1.image("images/thingsPeak.png", use_column_width=True)
    col2.image("images/elecNose.png", use_column_width=True)
    col2.markdown(
        """

### Introduction

This is our project that is designing Electronic Nose System for Gas Sensor Array data. Project contain developing full pipeline which contains these steps primarily

1. Collecting the data in the lab from sensors
2. Uploading the data on thingsPeak using RaspberryPi in real time
3. Analyzing data features using PCA, LDA and t-SNE plots etc.
4. Applying various Machine Learning approaches as well as Deep Learning algorithms on the data to classify the gases present in the mixture from sensors value
5. Using Regression based approaches predicting the concentration for the present gases in mixture.
6. Developing the API for automation of the whole process and real time visualization as well as prediction from the sensors value.

### Data Collection
Sensor data is collected from the sensors in the lab using RaspberryPi. Data is collected in real time and uploaded on the thingsPeak. Data is collected for 10 different gases and 10 different concentration for each gas. Data is collected for 2 different sensors. Then data was interpolated and preprocessed for further analysis.

### Applying Different Machine Learning Based Classifiers on Gas Sensor Dataset

In this project, for gas classification and predicting concentration of gases in mixture various supervised machine learning classifiers were applied and their performance were compared.


### Data Processing Workflow

On both datasets, PCA and t-SNE dimension reduction techniques were applied in order to plot and visualize the relationships between different attributes.The same workflow was followed and the same classifiers were applied on both of the datasets. I applied 10 classical classifiers( non-neural network based) and 1 keras based vanilla neural network classifier.

#### Classical Classifiers

1. K-Nearest Neighbor (KNN)
2. Support Vector Machine (SVM)
3. Gaussian Multinomial Naive Bayes (MultinomialNB)
4. Decision Tree
5. Random Forest
6. Extra Tree
7. Logistic Regression
8. KNN based Bagging
9. Logistic Regression
10. Majority Voting Ensemble Machine Learning Classifier

#### Regression Models

1. Linear Regression
2. Ridge Regression
3. Elastic Net
4. Support Vector Machine
5. SGD Regression

        """
    )

# mode = st.sidebar.selectbox("Select Mode", ["Training", "Testing"])


if mode == "Training":
    st.markdown("# Training")
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
            
        col1, col2 = st.columns(2)

        # Model selection
        model_type = col1.selectbox(
            "Select Model Type",
            (
                "Classification",
                "Regression",
            ),
        )

        if model_type == "Classification":
            model = col2.selectbox(
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
            model = col2.selectbox(
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
    st.markdown("# Testing")
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
    col1, col2 = st.columns(2)
    col1.subheader("Select Classifier")
    model = col1.selectbox(
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
    if col1.button("Predict Gas Type"):
        if model == "KNN":
            ans = load_model_and_predict(
                df,
                classes,
                "output/models/classification/KNeighborsClassifier.joblib",
                col1
            )
        elif model == "Naive Bayes":
            ans = load_model_and_predict(
                df, classes, "output/models/classification/GaussianNB.joblib",
                col1
            )

        elif model == "Logistic Regression":
            ans = load_model_and_predict(
                df,
                classes,
                "output/models/classification/LogisticRegression.joblib",
                col1
            )

        elif model == "Random Forest":
            ans = load_model_and_predict(
                df,
                classes,
                "output/models/classification/RandomForestClassifier.joblib",
                col1
            )

        elif model == "AdaBoost":
            ans = load_model_and_predict(
                df,
                classes,
                "output/models/classification/AdaBoostClassifier.joblib",
                col1
            )

        elif model == "VotingClassifier":
            ans = load_model_and_predict(
                df, classes, "output/models/classification/VotingClassifier.joblib",
                col1
            )

        elif model == "KNN with Bagging":
            ans = load_model_and_predict(
                df, classes, "output/models/classification/BaggingKNN.joblib",
                col1
            )

        elif model == "Decision Tree":
            ans = load_model_and_predict(
                df,
                classes,
                "output/models/classification/DecisionTreeClassifier.joblib",
                col1
            )

        elif model == "Extra Trees":
            ans = load_model_and_predict(
                df,
                classes,
                "output/models/classification/ExtraTreeClassifier.joblib",
                col1
            )

        elif model == "Linear SVC":
            ans = load_model_and_predict(
                df, classes, "output/models/classification/LinearSVC.joblib",
                col1
            )

    col2.subheader("Select Regressor")
    ans, name = load_model_and_predict(
        df, classes, "output/models/classification/KNeighborsClassifier.joblib", col2, print_ans=False
    )
    model = col2.selectbox(
        "Select a model",
        (
            "Linear Regression",
            "Ridge Regression",
            "Elastic Net",
            "Support Vector Machine",
            "SGD Regression",
        ),
    )
    if col2.button("Predict Gas Conc."):
        if model == "Linear Regression":
            if ans[0] < 3:
                load_model_and_predict_reg(
                    df,
                    f"output/models/regression/LinearRegression_{ans[0]}.joblib",
                    col2,
                    name,
                )
            else:
                name1 = name.split("+")[0]
                name2 = name.split("+")[1]
                load_model_and_predict_reg(
                    df,
                    f"output/models/regression/LinearRegression_{ans[0]}{0}.joblib",
                    col2,
                    name1,
                )
                load_model_and_predict_reg(
                    df,
                    f"output/models/regression/LinearRegression_{ans[0]}{1}.joblib",
                    col2,
                    name2,
                )

        elif model == "Ridge Regression":
            if ans[0] < 3:
                load_model_and_predict_reg(
                    df, f"output/models/regression/BayesianRidge_{ans[0]}.joblib", col2, name
                )
            else:
                name1 = name.split("+")[0]
                name2 = name.split("+")[1]
                load_model_and_predict_reg(
                    df,
                    f"output/models/regression/BayesianRidge_{ans[0]}{0}.joblib",
                    col2,
                    name1,
                )
                load_model_and_predict_reg(
                    df,
                    f"output/models/regression/BayesianRidge_{ans[0]}{1}.joblib",
                    col2,
                    name2,
                )

        elif model == "Elastic Net":
            if ans[0] < 3:
                load_model_and_predict_reg(
                    df, f"output/models/regression/ElasticNet_{ans[0]}.joblib", 
                    col2,
                    name,
                )
            else:
                name1 = name.split("+")[0]
                name2 = name.split("+")[1]
                load_model_and_predict_reg(
                    df, f"output/models/regression/ElasticNet_{ans[0]}{0}.joblib", col2, name1
                )
                load_model_and_predict_reg(
                    df, f"output/models/regression/ElasticNet_{ans[0]}{1}.joblib", col2, name2
                )

        elif model == "Support Vector Machine":
            if ans[0] < 3:
                load_model_and_predict_reg(
                    df, f"output/models/regression/SVMRegression_{ans[0]}.joblib", col2, name
                )
            else:
                name1 = name.split("+")[0]
                name2 = name.split("+")[1]
                load_model_and_predict_reg(
                    df,
                    f"output/models/regression/SVMRegression_{ans[0]}{0}.joblib",
                    col2,
                    name1,
                )
                load_model_and_predict_reg(
                    df,
                    f"output/models/regression/SVMRegression_{ans[0]}{1}.joblib",
                    col2,
                    name2,
                )

        elif model == "SGD Regression":
            if ans[0] < 3:
                load_model_and_predict_reg(
                    df, f"output/models/regression/SGDRegressor_{ans[0]}.joblib", col2, name
                )
            else:
                name1 = name.split("+")[0]
                name2 = name.split("+")[1]
                load_model_and_predict_reg(
                    df,
                    f"output/models/regression/SGDRegressor_{ans[0]}{0}.joblib",
                    col2,
                    name1
                )
                load_model_and_predict_reg(
                    df,
                    f"output/models/regression/SGDRegressor_{ans[0]}{1}.joblib",
                    col2,
                    name2
                )

if mode == "Contact Us":
    st.markdown("## Contact Us")
    st.markdown(
        """
        #### Team Members
        1. [Tushar Kumar](https://github.com/VincitoreSi)
        2. [Uday Bhanu](https://github.com/udaybhanu43)
        """
    )