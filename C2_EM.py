# 1.	Run the clustering algorithms on the datasets and describe what you see.

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture as gm
from matplotlib import pyplot as plt
import make_plots as myplots

from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.metrics import calinski_harabasz_score as chi
from sklearn.metrics import davies_bouldin_score as dbi

import itertools

from scipy import linalg
import matplotlib as mpl

from sklearn import mixture

from sklearn.manifold import TSNE

from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
import time
#from time import clock

import sys

#I think this was mentioned previously; measuring N-clusters against a variety of metrics and taking the plurality of
# N that maximizes those metrics is a valid way to choose clusters right?

# As long as all of those metrics are unsupervised, yes

from sklearn.base import ClusterMixin
from sklearn.mixture import GaussianMixture
from yellowbrick.cluster import KElbow
from yellowbrick.datasets import load_nfl

# https://stackoverflow.com/questions/58648969/the-supplied-model-is-not-a-clustering-estimator-in-yellowbrick
class GMClusters(GaussianMixture, ClusterMixin):

    def __init__(self, n_clusters=20, **kwargs):
        kwargs["n_components"] = n_clusters
        super(GMClusters, self).__init__(**kwargs)

    def fit(self, X):
        super(GMClusters, self).fit(X)
        self.labels_ = self.predict(X)
        return self


def em(dataX_train, datay_train, dataX_test, datay_test, timestr, k=9999, part='1', ds = 'Set1'):
    print("Part "+str(part))
    print(ds)
    print("expectation maximization")

    np.random.seed(42)

    # The GaussianMixture object implements the expectation-maximization (EM) algorithm for fitting mixture-of-Gaussian
    # models. It can also draw confidence ellipsoids for multivariate models, and compute the Bayesian Information
    # Criterion to assess the number of clusters in the data. A GaussianMixture.fit method is provided that learns a
    # Gaussian Mixture Model from train data. Given test data, it can assign to each sample the Gaussian it mostly
    # probably belong to using the GaussianMixture.predict method.

    # dataX_train = StandardScaler().fit_transform(dataX_train)
    em = gm(random_state=1)

    if k < 9999:
        em.set_params(n_components=k)
        em.fit(dataX_train)
        train_labels = em.predict(dataX_train)
        em.set_params(n_components=k)
        em.fit(dataX_test)
        test_labels = em.predict(dataX_test)
        return train_labels,test_labels #em.transform(dataX_train), em.transform(dataX_test)

    # km.set_params(n_clusters=k)
    # km.fit(dataX_train)
    # train_labels = km.labels_
    # km.fit(dataX_test)
    # test_labels = km.labels_
    # return train_labels, test_labels  # km.transform(dataX_train), km.transform(dataX_test)

    n_clusters =  [2,3,4,5,6,7,8,9,10,15,20]
    num_k = np.shape(n_clusters)
    fit_score = np.zeros(num_k[0])
    CHI = np.zeros(num_k[0])
    DBI = np.zeros(num_k[0])

    i=0
    for k in n_clusters:
        em.set_params(n_components=k)
        em.fit(dataX_train)
        # if k<10: myplots.visualize_3d(dataX_train, em.predict(dataX_train), algorithm="tsne", #"pca",
        #     title="part " + str(part) + ", " + str(ds) + ", " + str(k) + " clusters")
        if k<=20:
            if np.shape(dataX_train)[1] > 2:
                myplots.visualize_3d(dataX_train, em.predict(dataX_train), algorithm="tsne",  # "pca",
                                     title="part " + str(part) + ", " + str(ds) + ", 3D TSNE EM " + str(k) + " clusters",
                                     filename="part " + str(part) + ", " + str(ds) + ", 3D TSNE EM " + str(k) + " clusters" + timestr)

            plt1 = myplots.visualize_2d(dataX_train, em.predict(dataX_train), algorithm="tsne",  # "pca",
                                       title="part " + str(part) + ", " + str(ds) + ", EM, " + str(k) + " clusters")
            plt1.savefig("Part " + str(part) + ", " + str(ds) + ", 2D TSNE viz for EM with " + str(k) + " clusters" + timestr + '.png')
            plt1.close()
            # plt.show()

        fit_score[i] = em.score(dataX_train)
        CHI[i] = chi(dataX_train, em.predict(dataX_train))
        DBI[i] = dbi(dataX_train, em.predict(dataX_train))
        homo, comp, vmeas = myplots.compute_accuracy(datay_train, em.predict(dataX_train))
        print("homo,comp,vmeas for Part " + str(part) + ", " + ds + ", " + str(k) + " clusters, EM: " + str(homo) + ", "+ str(comp) + ", "+ str(vmeas))
        i+=1

    print(fit_score)


    # Elbow Method
    _, ax = plt.subplots()  # Create a new figure
    oz1 = KElbow(GMClusters(), k=(2, 30), force_model=True)
    oz1.fit(dataX_train,)
    oz1.show(outpath="Part " + str(part) + " " + ds + " Elbow Method for EM" + timestr + '.png')
    # oz.show()
    # oz.close()

    # Calinski Harabasz Score
    _, ax = plt.subplots()  # Create a new figure
    oz2 = KElbow(GMClusters(), k=(2, 30), metric='calinski_harabasz', force_model=True)
    oz2.fit(dataX_train)
    oz2.show(outpath="Part " + str(part) + " " + ds + " Calinski Harabasz Score for EM" + timestr + '.png')
    # oz.show()
    # oz.close()

    # Silhouette Score
    _, ax = plt.subplots()  # Create a new figure
    oz3 = KElbow(GMClusters(), k=(2, 30), metric='silhouette', force_model=True)
    oz3.fit(dataX_train)
    oz3.show(outpath="Part " + str(part) + " " + ds + " Silhouette Score for EM" + timestr + '.png')
    # oz.show()
    # oz.close()


    # Davies Bouldin score
    print(DBI)
    _, ax = plt.subplots()  # Create a new figure
    plt.plot(n_clusters, DBI, linestyle='--', marker='o', color='b')
    plt.xlabel('K')
    plt.ylabel('Davies Bouldin score')
    plt.title('Davies Bouldin score vs. K')
    plt.savefig("Part " + str(part) + " " + ds + " Davies Bouldin score for EM" + timestr + '.png')
    # plt.show()
    plt.close()

    # # Silhouette Score
    # visualizer = KElbowVisualizer(em, k=(2, 30), metric='silhouette', timings=True)
    # visualizer.fit(dataX_train)  # Fit the data to the visualizer
    # visualizer.show()  # Finalize and render the figure


    lowest_bic = np.infty
    bic = []
    n_components_range = range(1, 7)
    cv_types = ['spherical', 'tied', 'diag', 'full']
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a Gaussian mixture with EM
            gmm = mixture.GaussianMixture(n_components=n_components,
                                          covariance_type=cv_type)
            gmm.fit(dataX_train)
            bic.append(gmm.bic(dataX_train))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm

    bic = np.array(bic)
    color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                                  'darkorange'])
    clf = best_gmm
    bars = []

    # Plot the BIC scores
    plt.figure(figsize=(8, 6))
    spl = plt.subplot(2, 1, 1)
    for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
        xpos = np.array(n_components_range) + .2 * (i - 2)
        bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                      (i + 1) * len(n_components_range)],
                            width=.2, color=color))
    plt.xticks(n_components_range)
    plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
    plt.title('BIC score per model')
    xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 + \
           .2 * np.floor(bic.argmin() / len(n_components_range))
    plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
    spl.set_xlabel('Number of components')
    spl.legend([b[0] for b in bars], cv_types)

    # Plot the winner
    splot = plt.subplot(2, 1, 2)
    Y_ = clf.predict(dataX_train)
    for i, (mean, cov, color) in enumerate(zip(clf.means_, clf.covariances_,
                                               color_iter)):
        v, w = linalg.eigh(cov)
        if not np.any(Y_ == i):
            continue
        plt.scatter(dataX_train[Y_ == i, 0], dataX_train[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan2(w[0][1], w[0][0])
        angle = 180. * angle / np.pi  # convert to degrees
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(.5)
        splot.add_artist(ell)

    plt.xticks(())
    plt.yticks(())
    plt.title(f'Selected GMM: {best_gmm.covariance_type} model, '
              f'{best_gmm.n_components} components')
    plt.subplots_adjust(hspace=.35, bottom=.02)
    plt.savefig("Part " + str(part) + " " + ds + " BIC Score for EM" + timestr + '.png')
    # plt.show()
    plt.close()









# For EM, I am exploring BIC score. Are there any other metrics that I can explore?
# BIC metric is not giving good results for one of my datasets
#
# Michael Driscoll:classical_building:  2 days ago
# For EM you have BIC using all four covariance types, you have DBS, CHI, and Silhouette
#
# Taneem  2 days ago
# @Michael Driscoll which one is CHI? Clarinski or something else?
#
# Michael Driscoll:classical_building:  2 days ago
# Calinski-Harabasz Index, CHI because no way am I ever going to remember that to type out
#
# Mark  1 day ago
# @Michael Driscoll what is DBS? I'm only seeing DBScan. Is it that the same thing?
#
# Michael Driscoll:classical_building:  1 day ago
# Davies-Bouldin Index Score, take a look at the scikit clustering performance evaluation section
# :+1::skin-tone-2:
# 1
#
#
# Taneem  1 day ago
# yeah I used all 4 of them for the kmeans , k selection
#
# Taneem  1 day ago
# unfortunately they are all over the map
#
# Mark  1 day ago
# @Taneem how did you decide which to use, then?
#
# Taneem  1 day ago
# I picked the one that had majority votes and then also looked silhouette visualizer
#
# Taneem  1 day ago
# like for example if both k=2 and k-4 were given by the various metrics, then I compared the solhouette visualizer
# to see which one looked better (you can tell based on individual cluster size and score)
#
# Mark  1 day ago
# Cool. Thanks!
# :+1:
# 1
#
#
# Jeremy  1 day ago
# @Taneem how did you use SilhouetteVisualizer for EM? It doesn't work for me bc its a mixture estimator not a
# clustering estimator
#
# Taneem  1 day ago
# https://stackoverflow.com/questions/58648969/the-supplied-model-is-not-a-clustering-estimator-in-yellowbrick
# Stack OverflowStack Overflow
# The supplied model is not a clustering estimator in YellowBrick
# I am trying to visualize an elbow plot for my data using YellowBrick's KElbowVisualizer and SKLearn's
# Expectation Maximization algorithm class: GaussianMixture. When I run this, I get the error in...
#
# Jeremy  1 day ago
# @Taneem thank you. This doesn't seem to work for me for the SilhouetteVisualizer. how can I set the covariance_type
# and random_state with this?   I tried clusterer = GMClusters(k,**{'covariance_type':cov,'random_state':SEED})     b
# ut then .fit(X) method fails