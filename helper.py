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
    # save in csv file
    datatrain1.to_csv("Data/data/ethylene_CO.csv", index=False)
    datatrain2.to_csv("Data/data/ethylene_methane.csv", index=False)
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

def load_model_and_predict(df, classes, path, stc, print_ans=True):
    """Loads a pre-trained model and uses it to make predictions on the provided data frame.

    Args:
      df: Pandas DataFrame containing the input data to make predictions on.
      classes: List of class name strings corresponding to the model's numeric outputs.
      path: File path to the pre-trained model file.
      stc: Streamlit component to print the results to.
      print_ans: Whether to print the prediction result to stc.

    Returns:
      ans: NumPy array of predicted class numbers.
      name: Class name corresponding to the predicted class number.
    """
    model = load(path)
    ans = model.predict(df)
    # get class name from class number
    name = classes[ans[0]]
    if print_ans:
        stc.markdown(f"## {name}")
    return ans, name

def load_model_and_predict_reg(df, path, stc,  i=""):
    model = load(path)
    ans = model.predict(df)
    # stc.write(f"Predicted {i} Gas Concentration: {ans[0]} ppm")
    stc.metric(label=f"{i}", value=f"{ans[0]:.3f} ppm")
    return ans

def txt_to_csv():
    """
    Load and prepare data
    """
    # Load data
    print("Loading data...")
    # columns = ["Time (seconds)", "Methane conc (ppm)", "Ethylene conc (ppm)"] + ["Sensor " + str(i) for i in range(1, 17)]
    # df = pd.read_csv(PATH3, delim_whitespace=True, names=columns)
    datatrain1 = pd.read_csv("Data/data/ethylene_CO.txt", delim_whitespace=True)
    datatrain2 = pd.read_csv("Data/data/ethylene_methane.txt", delim_whitespace=True)
    # save in csv file
    datatrain1.to_csv("Data/data/ethylene_CO.csv", index=False)
    datatrain2.to_csv("Data/data/ethylene_methane.csv", index=False)

def new_data(path):
    print("Loading data...")
    data = pd.read_excel(path)
    # I have this data in the given form and I need to convert this dataframe such that it will have time, sensor 1 and sensor 2 , Gas Conc. and GasType columns and for each gas label there will be sensors values and gas label. How to transform given data in this form. Write code for this.
    # new df will have columns: Time, Sensor 1, Sensor 2, Gas Conc., GasType
    df = pd.DataFrame(columns=["Time", "Sensor 1", "Sensor 2", "Gas Conc.", "GasType"])
    
    print(data.head())

def prepare(path):
    df = pd.read_csv(path)
    # drop last 10 columns
    df = df.iloc[:, :-11]
    # save csv
    df.to_csv("Data/data/ethylene_methane_5.csv", index=False)

def send_data_to_thingsPeak(path):
    # write code to send data to thingsPeak
    df = pd.read_csv(path)
    channelID = "2395244"
    writeAPIKey = "HLXRHH1FDD3SEVJQ"
    channelWriteURL = "https://api.thingspeak.com/update?api_key="
    for idx, col in enumerate(df.columns):
        if col != "Time (seconds)":
            data = df[col].values
            for i in range(len(data)):
                url = (
                    channelWriteURL
                    + writeAPIKey
                    + "&field"
                    + str(idx)
                    + "="
                    + str(data[i])
                )
                response = requests.get(url)
                print(response.status_code)
        print("---------")

    print("Data sent to thingsPeak")

if __name__ == "__main__":
    # load_gas_data_for_regression("Data/data/expanded_data.csv", 3)
    # new_data("Data/data/time_series_data.xlsx")
    # prepare("Data/data/ethylene_methane.csv")
    send_data_to_thingsPeak("Data/data/ethylene_methane_5.csv")