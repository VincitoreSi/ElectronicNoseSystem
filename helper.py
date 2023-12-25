"""
file: helper.py
author: @VincitoreSi
date: 2023-12-16
brief: Helper functions for the project
"""

from dependencies.dependencies import *


def lead_and_prepare_data():
    """
    Load and prepare data
    """
    # Load data
    print("Loading data...")
    # columns = ["Time (seconds)", "Methane conc (ppm)", "Ethylene conc (ppm)"] + ["Sensor " + str(i) for i in range(1, 17)]
    # df = pd.read_csv(PATH3, delim_whitespace=True, names=columns)
    datatrain1 = pd.read_csv("Data/data/ethylene_CO.txt", delim_whitespace=True)
    datatrain2 = pd.read_csv("Data/data/ethylene_methane.txt", delim_whitespace=True)
    X1 = np.array(datatrain1)
    X2 = np.array(datatrain2)
    print(X1.shape)
    print(X2.shape)

    array_list = [X1, X2]
    sample = np.concatenate([X1, X2])
    lengths = [len(X1), len(X2)]
    datatrain_array = np.vstack(array_list)

    xtrain = datatrain_array

    # Setting the target value 0 for ethylene_CO and 1 for ethylene_methane
    y1 = np.zeros(4208261)
    y2 = np.ones(4178504)
    ytrain = np.concatenate([y1, y2])

    print(f"X_train shape: {xtrain.shape}")
    print(f"y_train shape: {ytrain.shape}")

    max_abs_scaler = MaxAbsScaler()
    xtrain = max_abs_scaler.fit_transform(xtrain)
    X_train, X_test, y_train, y_test = train_test_split(
        xtrain, ytrain, test_size=0.001, random_state=1
    )
    return X_train, X_test, y_train, y_test


def plot_confusion_matrix(
    cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    print(cm)

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")


def load_gas_data(path):
    print("Loading data...")
    data = pd.read_csv(path)
    X = data.iloc[:, :-2]
    y = data.iloc[:, -2]
    print(f"X shape: {X.shape}\ny shape: {y.shape}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1
    )
    return X_train, X_test, y_train, y_test


def preprocess_data_clf(df):
    X = df.iloc[:, :-2]
    y = df.iloc[:, -2]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=1
    )
    return X_train, X_test, y_train, y_test

def preprocess_data_reg(data, cls):
    data_cls = data[data["GasType"] == cls]
    if cls == 3:
        data_cls = data_cls.drop(columns=["GasType"], axis=1)
        data_cls['ppm1'] = data_cls['ppm'].apply(lambda x: float(x.split('+')[0]))
        data_cls['ppm2'] = data_cls['ppm'].apply(lambda x: float(x.split('+')[1]))
        data_cls = data_cls.drop(columns=["ppm"], axis=1)
        data_cls = data_cls.reset_index(drop=True)
        X = data_cls.iloc[:, :-2]
        y = data_cls.iloc[:, -2:]
    else:
        data_cls = data_cls.drop(columns=["GasType"], axis=1)
        data_cls = data_cls.reset_index(drop=True)
        X = data_cls.iloc[:, :-1]
        y = data_cls.iloc[:, -1]
    print(data_cls.head())
    print(f"X shape: {X.shape}\ny shape: {y.shape}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=1
    )
    return X_train, X_test, y_train, y_test

def load_gas_data_for_regression(path, cls):
    print("Loading data...")
    data = pd.read_csv(path)
    data_cls = data[data["GasType"] == cls]
    if cls == 3:
        # then make extra column for gas conc. separating from '+' in conc. column
        data_cls = data_cls.drop(columns=["GasType"], axis=1)
        data_cls['ppm1'] = data_cls['ppm'].apply(lambda x: float(x.split('+')[0]))
        data_cls['ppm2'] = data_cls['ppm'].apply(lambda x: float(x.split('+')[1]))
        data_cls = data_cls.drop(columns=["ppm"], axis=1)
        data_cls = data_cls.reset_index(drop=True)
        X = data_cls.iloc[:, :-2]
        y = data_cls.iloc[:, -2:]
    else:
        data_cls = data_cls.drop(columns=["GasType"], axis=1)
        data_cls = data_cls.reset_index(drop=True)
        X = data_cls.iloc[:, :-1]
        y = data_cls.iloc[:, -1]
    print(data_cls.head())
    print(f"X shape: {X.shape}\ny shape: {y.shape}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=1
    )
    return X_train, X_test, y_train, y_test

# Function to download model
def get_download_link(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{file_path}">Download Trained Model</a>'
    return href

if __name__ == "__main__":
    load_gas_data_for_regression("Data/data/expanded_data.csv", 3)