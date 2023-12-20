"""
file: helper.py
author: @VincitoreSi
date: 2023-12-16
brief: Helper functions for the project
"""

from dependencies.dependencies import plt, itertools, np, pd, confusion_matrix, MaxAbsScaler, train_test_split

def lead_and_prepare_data():
    """
    Load and prepare data 
    """
    # Load data
    print("Loading data...")
    # columns = ["Time (seconds)", "Methane conc (ppm)", "Ethylene conc (ppm)"] + ["Sensor " + str(i) for i in range(1, 17)]
    # df = pd.read_csv(PATH3, delim_whitespace=True, names=columns)
    datatrain1=pd.read_csv('Data/data/ethylene_CO.txt', delim_whitespace=True)
    datatrain2=pd.read_csv('Data/data/ethylene_methane.txt', delim_whitespace=True)
    X1=np.array(datatrain1)
    X2=np.array(datatrain2)
    print(X1.shape)
    print(X2.shape)

    array_list=[X1,X2]
    sample = np.concatenate([X1, X2])
    lengths = [len(X1), len(X2)]
    datatrain_array=np.vstack(array_list)
    
    xtrain = datatrain_array

    #Setting the target value 0 for ethylene_CO and 1 for ethylene_methane
    y1=np.zeros(4208261)
    y2=np.ones(4178504)
    ytrain=np.concatenate([y1,y2])
    
    print(f"X_train shape: {xtrain.shape}")
    print(f"y_train shape: {ytrain.shape}")

    max_abs_scaler = MaxAbsScaler()
    xtrain = max_abs_scaler.fit_transform(xtrain) 
    X_train, X_test, y_train, y_test = train_test_split(xtrain, ytrain, test_size=.001,random_state=1)
    return X_train, X_test, y_train, y_test

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def load_gas_data(path):
    data = pd.read_csv(path)
    X = data.iloc[:, :-2]
    y = data.iloc[:, -2]
    return np.array(X), np.array(y)