#checking in to see if I'm missing anything, here are all my plots:

# -Tuning plots or a table for my clustering
# and DR algs

# -Silhouette coefficient + time for clustering,
#     cumulative explained variance ratio >= 95% (PCA),
#     Mean kurtosis (ICA),
#     Reconstruction MSE (RP),
#     f1 score for my own choice

# -TSNE plots and Pair Confusion matrices for initial datasets before clustering,
#      datasets after clustering,
#      ***and datasets after dimensionality reduction

# -learning curves (score vs. N, time vs. N, score vs time) for neural network
#  (all three of preprocessed,post-clustering, post-DR)


from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import plotly
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import time
import seaborn as sns

import sklearn as sk
import sklearn.model_selection as ms
from sklearn.model_selection import learning_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
import plotly
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
#from time import clock
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import ShuffleSplit
from sklearn import metrics


def compute_accuracy(correctY,clusters):
    homo = metrics.homogeneity_score(correctY, clusters)
    comp = metrics.completeness_score(correctY, clusters)
    vmeas = metrics.v_measure_score(correctY, clusters)
    return homo,comp,vmeas


def visualize_3d(X, y, algorithm="tsne", title="Data in 3D",filename="test.html"):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    Xorig=X

    if algorithm == "tsne":
        reducer = TSNE(n_components=3, random_state=47, n_iter=400, angle=0.6)
    elif algorithm == "pca":
        reducer = PCA(n_components=3, random_state=47)
    else:
        raise ValueError("Unsupported dimensionality reduction algorithm given.")

    if X.shape[1] > 3:
        X = reducer.fit_transform(X)
    else:
        if type(X) == pd.DataFrame:
            X = X.values

    marker_shapes = ["circle", "diamond", "circle-open", "square", "diamond-open", "cross", "square-open",
                     "circle", "diamond", "circle-open", "square", "diamond-open", "cross", "square-open",
                     "circle", "diamond", "circle-open", "square", "diamond-open", "cross", "square-open",
                     "circle", "diamond", "circle-open", "square", "diamond-open", "cross", "square-open",
                     "circle", "diamond", "circle-open", "square", "diamond-open", "cross", "square-open",
                     "circle", "diamond", "circle-open", "square", "diamond-open", "cross", "square-open",
                     "circle", "diamond", "circle-open", "square", "diamond-open", "cross", "square-open",
                     "circle", "diamond", "circle-open", "square", "diamond-open", "cross", "square-open",
                     "circle", "diamond", "circle-open", "square", "diamond-open", "cross", "square-open",
                     "circle", "diamond", "circle-open", "square", "diamond-open", "cross", "square-open",
                     "circle", "diamond", "circle-open", "square", "diamond-open", "cross", "square-open",]
    traces = []
    for hue in np.unique(y):
        X1 = X[y == hue]

        trace = go.Scatter3d(
            x=X1[:, 0],
            y=X1[:, 1],
            z=X1[:, 2],
            mode='markers',
            name=str(hue),
            marker=dict(
                size=12,
                symbol=marker_shapes.pop(),
                line=dict(
                    width=int(np.random.randint(3, 10) / 10)
                ),
                opacity=int(np.random.randint(6, 10) / 10)
            )
        )
        traces.append(trace)

    layout = go.Layout(
        title=title,
        scene=dict(
            xaxis=dict(
                title='Dim 1'),
            yaxis=dict(
                title='Dim 2'),
            zaxis=dict(
                title='Dim 3'), ),
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        )
    )
    fig = go.Figure(data=traces, layout=layout)
    plotly.io.write_image(fig, filename+".png", format=None,scale=None, width=None, height=None)
    #plot(fig, show_link=True, filename=filename+".html")
    #plot(fig)

def visualize_3d1(X, y, algorithm="tsne", title="Data in 3D",filename="test.html"):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    if algorithm == "tsne":
        reducer = TSNE(n_components=3, random_state=47, n_iter=400, angle=0.6)
    elif algorithm == "pca":
        reducer = PCA(n_components=3, random_state=47)
    else:
        raise ValueError("Unsupported dimensionality reduction algorithm given.")

    # if X.shape[1] > 3:
    #     X = reducer.fit_transform(X)
    # else:
    #     if type(X) == pd.DataFrame:
    #         X = X.values

    marker_shapes = ["circle", "diamond", "circle-open", "square", "diamond-open", "cross", "square-open",
                     "circle", "diamond", "circle-open", "square", "diamond-open", "cross", "square-open",
                     "circle", "diamond", "circle-open", "square", "diamond-open", "cross", "square-open",
                     "circle", "diamond", "circle-open", "square", "diamond-open", "cross", "square-open",
                     "circle", "diamond", "circle-open", "square", "diamond-open", "cross", "square-open",
                     "circle", "diamond", "circle-open", "square", "diamond-open", "cross", "square-open",
                     "circle", "diamond", "circle-open", "square", "diamond-open", "cross", "square-open",
                     "circle", "diamond", "circle-open", "square", "diamond-open", "cross", "square-open",
                     "circle", "diamond", "circle-open", "square", "diamond-open", "cross", "square-open",
                     "circle", "diamond", "circle-open", "square", "diamond-open", "cross", "square-open",]
    traces = []
    for hue in np.unique(y):
        X1 = X[y == hue]

        trace = go.Scatter3d(
            x=X1[:, 2],
            y=X1[:, 3],
            z=X1[:, 4],
            mode='markers',
            name=str(hue),
            marker=dict(
                size=12,
                symbol=marker_shapes.pop(),
                line=dict(
                    width=int(np.random.randint(3, 10) / 10)
                ),
                opacity=int(np.random.randint(6, 10) / 10)
            )
        )
        traces.append(trace)

    layout = go.Layout(
        title=title,
        scene=dict(
            xaxis=dict(
                title='Dim 1'),
            yaxis=dict(
                title='Dim 2'),
            zaxis=dict(
                title='Dim 3'), ),
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        )
    )
    fig = go.Figure(data=traces, layout=layout)
    #fig.savefig('Data' + timestr + '.png')
    plot(fig, show_link=True, filename=filename)
    #plot(fig)

def visualize_3d2(X, y, algorithm="tsne", title="Data in 3D",filename="test.html"):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    if algorithm == "tsne":
        reducer = TSNE(n_components=3, random_state=47, n_iter=400, angle=0.6)
    elif algorithm == "pca":
        reducer = PCA(n_components=3, random_state=47)
    else:
        raise ValueError("Unsupported dimensionality reduction algorithm given.")

    # if X.shape[1] > 3:
    #     X = reducer.fit_transform(X)
    # else:
    #     if type(X) == pd.DataFrame:
    #         X = X.values

    marker_shapes = ["circle", "diamond", "circle-open", "square", "diamond-open", "cross", "square-open",
                     "circle", "diamond", "circle-open", "square", "diamond-open", "cross", "square-open",
                     "circle", "diamond", "circle-open", "square", "diamond-open", "cross", "square-open",
                     "circle", "diamond", "circle-open", "square", "diamond-open", "cross", "square-open",
                     "circle", "diamond", "circle-open", "square", "diamond-open", "cross", "square-open",
                     "circle", "diamond", "circle-open", "square", "diamond-open", "cross", "square-open",
                     "circle", "diamond", "circle-open", "square", "diamond-open", "cross", "square-open",
                     "circle", "diamond", "circle-open", "square", "diamond-open", "cross", "square-open",
                     "circle", "diamond", "circle-open", "square", "diamond-open", "cross", "square-open",
                     "circle", "diamond", "circle-open", "square", "diamond-open", "cross", "square-open", ]
    traces = []
    for hue in np.unique(y):
        X1 = X[y == hue]

        trace = go.Scatter3d(
            x=X1[:, 6],
            y=X1[:, 7],
            z=X1[:, 8],
            mode='markers',
            name=str(hue),
            marker=dict(
                size=12,
                symbol=marker_shapes.pop(),
                line=dict(
                    width=int(np.random.randint(3, 10) / 10)
                ),
                opacity=int(np.random.randint(6, 10) / 10)
            )
        )
        traces.append(trace)

    layout = go.Layout(
        title=title,
        scene=dict(
            xaxis=dict(
                title='Dim 7'),
            yaxis=dict(
                title='Dim 8'),
            zaxis=dict(
                title='Dim 9'), ),
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        )
    )
    fig = go.Figure(data=traces, layout=layout)
    #fig.savefig('Data' + timestr + '.png')
    plot(fig, show_link=True, filename=filename)
    #plot(fig)

def visualize_2d(X,y,algorithm="tsne",title="Data in 2D",figsize=(8,8)):
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    if algorithm=="tsne":
        reducer = TSNE(n_components=2,random_state=47,n_iter=400,angle=0.6)
    elif algorithm=="pca":
        reducer = PCA(n_components=2,random_state=47)
    else:
        raise ValueError("Unsupported dimensionality reduction algorithm given.")
    # if X.shape[1]>2:
    #     X = reducer.fit_transform(X)
    # else:
    #     if type(X)==pd.DataFrame:
    #     	X=X.values
    f, (ax1) = plt.subplots(nrows=1, ncols=1,figsize=figsize)
    sns.scatterplot(x=X[:,0],y=X[:,1],hue=y,ax=ax1)
    ax1.set_title(title)
    return plt


def plot_loss_curve2(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    #FROM: https: // scikit - learn.org / stable / auto_examples / model_selection / plot_learning_curve.html
    """
    """
    if axes is None:
        #_, axes = plt.subplots(1, 3, figsize=(20, 5))
        _, axes = plt.subplots(1, 2, figsize=(20, 5))

    X_train, X_VAL, y_train, y_VAL = sk.model_selection.train_test_split(X, y, test_size=0.3,
                                                                                           random_state=42,
                                                                                           stratify=y)

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Iterations")
    axes[0].set_ylabel("Loss")

    estimator.fit(X_train,y_train)
    fit_losscurve = estimator['nn'].loss_curve_

    estimator.score(X_train, y_train)


    # train_sizes, train_scores, test_scores, fit_times, _ = \
    #     learning_curve(estimator, X, y, scoring = 'accuracy', cv=cv, n_jobs=n_jobs,
    #                    train_sizes=train_sizes,
    #                    return_times=True)
    #
    # train_scores_mean = np.mean(train_scores, axis=1)
    # train_scores_std = np.std(train_scores, axis=1)
    # test_scores_mean = np.mean(test_scores, axis=1)
    # test_scores_std = np.std(test_scores, axis=1)
    # fit_times_mean = np.mean(fit_times, axis=1)
    # fit_times_std = np.std(fit_times, axis=1)

    # Plot loss curve
    axes[0].grid()
    # axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
    #                      train_scores_mean + train_scores_std, alpha=0.1,
    #                      color="r")
    # axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
    #                      test_scores_mean + test_scores_std, alpha=0.1,
    #                      color="g")

    axes[0].plot(estimator['nn'].loss_curve_)

    # axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
    #              label="Training score")
    # axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
    #              label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    # axes[1].grid()
    # axes[1].plot(train_sizes, fit_times_mean, 'o-')
    # axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
    #                      fit_times_mean + fit_times_std, alpha=0.1)
    # axes[1].set_xlabel("Training examples")
    # axes[1].set_ylabel("fit_times")
    # axes[1].set_title("Scalability of the model")

    # # Plot fit_time vs score
    # axes[2].grid()
    # axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    # axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
    #                      test_scores_mean + test_scores_std, alpha=0.1)
    # axes[2].set_xlabel("fit_times")
    # axes[2].set_ylabel("Score")
    # axes[2].set_title("Performance of the model")

    return plt

def plot_loss_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    #FROM: https: // scikit - learn.org / stable / auto_examples / model_selection / plot_learning_curve.html
    """
    """
    if axes is None:
        # _, axes = plt.subplots(1, 3, figsize=(20, 5))
        _, axes = plt.subplots(1, 2, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Iterations")
    axes[0].set_ylabel("Loss")

    # train_sizes, train_scores, test_scores, fit_times, _ = \
    #     learning_curve(estimator, X, y, scoring = 'accuracy', cv=cv, n_jobs=n_jobs,
    #                    train_sizes=train_sizes,
    #                    return_times=True)
    #
    # train_scores_mean = np.mean(train_scores, axis=1)
    # train_scores_std = np.std(train_scores, axis=1)
    # test_scores_mean = np.mean(test_scores, axis=1)
    # test_scores_std = np.std(test_scores, axis=1)
    # fit_times_mean = np.mean(fit_times, axis=1)
    # fit_times_std = np.std(fit_times, axis=1)

    # Plot loss curve
    axes[0].grid()
    # axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
    #                      train_scores_mean + train_scores_std, alpha=0.1,
    #                      color="r")
    # axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
    #                      test_scores_mean + test_scores_std, alpha=0.1,
    #                      color="g")

    axes[0].plot(estimator['nn'].loss_curve_)

    # axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
    #              label="Training score")
    # axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
    #              label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    # axes[1].grid()
    # axes[1].plot(train_sizes, fit_times_mean, 'o-')
    # axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
    #                      fit_times_mean + fit_times_std, alpha=0.1)
    # axes[1].set_xlabel("Training examples")
    # axes[1].set_ylabel("fit_times")
    # axes[1].set_title("Scalability of the model")

    # # Plot fit_time vs score
    # axes[2].grid()
    # axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    # axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
    #                      test_scores_mean + test_scores_std, alpha=0.1)
    # axes[2].set_xlabel("fit_times")
    # axes[2].set_ylabel("Score")
    # axes[2].set_title("Performance of the model")

    return plt


def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    #FROM: https: // scikit - learn.org / stable / auto_examples / model_selection / plot_learning_curve.html
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
        # _, axes = plt.subplots(1, 3, figsize=(20, 5))
        _, axes = plt.subplots(1, 2, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(estimator, X, y, scoring = 'accuracy', cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve

    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # # Plot fit_time vs score
    # axes[2].grid()
    # axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    # axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
    #                      test_scores_mean + test_scores_std, alpha=0.1)
    # axes[2].set_xlabel("fit_times")
    # axes[2].set_ylabel("Score")
    # axes[2].set_title("Performance of the model")

    return plt





