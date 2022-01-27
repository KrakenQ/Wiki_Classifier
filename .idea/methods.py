import matplotlib.pyplot as plt
import numpy as np
import seaborn as seaborn
from sklearn.model_selection import learning_curve
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

"""
In the first column, first row the learning curve (accuracy).
The plots in the second row show the times required by the models to train with various sizes of training dataset. 
The plots in the third row show how much time was required to train the models for each training sizes.
"""
def plot_learning_curve(estimator,
                        title,
                        X,
                        y,
                        axes=None,
                        ylim=None,
                        cv=None,
                        n_jobs=None,
                        train_sizes=np.linspace(0.1, 1.0, 5),
                        ):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.
    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.
    title : str
        Title for the chart.
    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.
    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.
    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.
    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).
    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.
        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.
        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.
    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
        )
    axes[0].fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
        )
    axes[0].plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    axes[0].plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, "o-")
    axes[1].fill_between(
        train_sizes,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1,
        )
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, "o-")
    axes[2].fill_between(
        fit_times_mean,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        )
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt

def print_graphs(estimator, X, y, predicted, estimator_name, categories_dict):
    from sklearn.metrics import plot_confusion_matrix, confusion_matrix
    cm = confusion_matrix(y, predicted)
    print(cm)
    draw_confusion_matrix_heatmap(cf_matrix=cm, title=estimator_name + " confusion matrix", xlabels=list(categories_dict.values()), ylabels=list(categories_dict.values()))
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    #fig.tight_layout()
    title = r"Learning Curves " + estimator_name
    plot_learning_curve(
        estimator, title, X, y, axes=axes, ylim=(0.0, 1.01), n_jobs=-1
    )
    plt.show()
