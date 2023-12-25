"""
file: pca.py
author: @VincitoreSi
date: 2023-12-16
brief: apply pca to the data
"""

from dependencies.dependencies import *
from helper import lead_and_prepare_data, load_gas_data


def pca(X_train, n_comp):
    xtrain = PCA(n_components=n_comp).fit_transform(X_train)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(
        xtrain[0:4208261, 0],
        xtrain[0:4208261, 1],
        xtrain[0:4208261, 2],
        markersize=8,
        label="ethylene_CO",
    )
    ax.plot(
        xtrain[1048575:8386765, 0],
        xtrain[1048575:8386765, 1],
        xtrain[1048575:8386765, 2],
        markersize=8,
        label="ethylene_methane",
    )

    # label the axes
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("PCA on gas sensor binary dataset")
    ax.legend(loc="upper right")

    # show plot
    # plt.show()
    # save plot to file
    plt.savefig("output/pca2.png")


# function to fit PCA on data
def apply_pca(data, n_components):
    pca = PCA(n_components=n_components)
    pca.fit(data)
    return pca


# function to get the PCA transformed data
def get_pca_data(pca, data):
    return pd.DataFrame(pca.transform(data), columns=["PC1", "PC2"])


# visualize the data to know if it is separable or not
def visualize_data(data, labels, title, ax_labels, legend= ["Gas 1: Ethylene", "Gas 2: Acetone", "Mixture: Ethylene + Acetone"]):
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = {"tab:blue": 1, "tab:orange": 2, "tab:green": 3}
    for color in colors.keys():
        ax.scatter(
            data.loc[labels == colors[color], ax_labels[0]],
            data.loc[labels == colors[color], ax_labels[1]],
            c=color,
            s=50,
        )
        ax.legend(legend)
        ax.grid()
        ax.set_xlabel(ax_labels[0])
        ax.set_ylabel(ax_labels[1])
        ax.set_title(title)
    plt.savefig(f"output/images/visualization/{title}.png")
    # plt.show()


def apply_lda(data, labels, n_components):
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    lda.fit(data, labels)
    transformed_data = lda.transform(data)
    return pd.DataFrame(transformed_data, columns=["LD1", "LD2"])


def apply_tsne(data, n_components):
    tsne = TSNE(n_components=n_components)
    tsne.fit(data)
    transformed_data = tsne.fit_transform(data)
    return pd.DataFrame(transformed_data, columns=["TSNE1", "TSNE2"])


def main():
    """
    Main function
    """
    # X_train, X_test, y_train, y_test = lead_and_prepare_data()
    X, X_test, y, y_test = load_gas_data("Data/data/expanded_data.csv")
    X, y = np.array(X), np.array(y)
    pc = apply_pca(X, 2)
    pc_data = get_pca_data(pc, X)
    visualize_data(pc_data, y, "PCA_on_gas_sensor_binary_dataset", ["PC1", "PC2"])
    ld = apply_lda(X, y, 2)
    visualize_data(ld, y, "LDA_on_gas_sensor_binary_dataset", ["LD1", "LD2"])
    tsne = apply_tsne(X, 2)
    visualize_data(tsne, y, "TSNE_on_gas_sensor_binary_dataset", ["TSNE1", "TSNE2"])


if __name__ == "__main__":
    main()
