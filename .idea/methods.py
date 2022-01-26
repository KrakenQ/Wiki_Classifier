import matplotlib.pyplot as plt
import numpy as np
import seaborn as seaborn

def draw_confusion_matrix_heatmap(cf_matrix, title, xlabels, ylabels, remove_zeros=True, remove_diag=True):
    plt.figure()
    plt.title(title)
    # mask = np.diag(cf_matrix.shape, k=0)
    if remove_diag:
        np.fill_diagonal(cf_matrix, 0)
    if remove_zeros:
        rows_to_remove = []
        for i, row in enumerate(cf_matrix):
            if np.all(row == 0):
                rows_to_remove.append(i)
        new_xlabels = np.delete(xlabels, rows_to_remove)
        new_matrix = np.delete(cf_matrix, rows_to_remove, axis=0)

        cols_to_remove = []
        for i, col in enumerate(new_matrix.T):
            if np.all(col == 0):
                cols_to_remove.append(i)
        new_ylabels = np.delete(ylabels, cols_to_remove)
        new_matrix = np.delete(new_matrix, cols_to_remove, axis=1)

        new_matrix = new_matrix.astype(float)
        new_matrix[new_matrix == 0] = np.nan
    else:
        new_matrix = cf_matrix
        new_xlabels = xlabels
        new_ylabels = ylabels

    ax = seaborn.heatmap(new_matrix, yticklabels=new_xlabels, xticklabels=new_ylabels, annot=True, linewidth=.5, linecolor="black", cmap="BuPu")
    ax.set(xlabel="Predicted category", ylabel="Actual category")
    plt.show()
