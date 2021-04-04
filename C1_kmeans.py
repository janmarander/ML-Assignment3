# 1.	Run the clustering algorithms on the datasets and describe what you see.
# https://towardsdatascience.com/cheat-sheet-to-implementing-7-methods-for-selecting-optimal-number-of-clusters-in-python-898241e1d6ad


import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans as kmeans
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import make_plots as myplots
import matplotlib.cm as cm

import itertools

from scipy import linalg
import matplotlib as mpl

from sklearn import mixture

from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.metrics import calinski_harabasz_score as chi
from sklearn.metrics import davies_bouldin_score as dbi
from yellowbrick.cluster import KElbowVisualizer

from sklearn.metrics import silhouette_samples, silhouette_score

from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
import time
#from time import clock

import sys

# 1. ***Run the clustering algorithms on the datasets and describe what you see.
# 2.	Apply the dimensionality reduction algorithms to the two datasets and describe what you see.
# 3. ***Reproduce your clustering experiments, but on the data after you've run dimensionality reduction on it. Yes,
# that’s 16 combinations of datasets, dimensionality reduction, and clustering method. You should look at all of them,
# but focus on the more interesting findings in your report.
# 4.	Apply the dimensionality reduction algorithms to one of your datasets from assignment #1 (if you've reused the
# datasets from assignment #1 to do experiments 1-3 above then you've already done this) and rerun your neural network
# learner on the newly projected data.
# 5. ***Apply the clustering algorithms to the same dataset to which you just applied the dimensionality reduction
# algorithms (you've probably already done this), treating the clusters as if they were new features. In other words,
# treat the clustering algorithms as if they were dimensionality reduction algorithms. Again, rerun your neural network
# learner on the newly projected data.

# In part 4 & 5, do we need to run the neural network with different number of clusters and components? Or is it
# sufficient to run only with a number of clusters / components we selected as best in part 1 & 2 and compare with
# the base case? (A) Run with the optimal K clusters and N components you found.


#I think this was mentioned previously; measuring N-clusters against a variety of metrics and taking the plurality of
# N that maximizes those metrics is a valid way to choose clusters right?
# As long as all of those metrics are unsupervised, yes


# for k-means we have yellowbricks which can be used to find best k using different scores (distortion, silhouette, calinski).

def km(dataX_train, datay_train, dataX_test, datay_test, timestr, k=9999, part='1', ds = 'Set1'):
    print("Part "+str(part))
    print(ds)
    print("Kmeans")

    np.random.seed(42)

    # # Setting up the scaling pipeline
    # pipeline_order = [('scaler', StandardScaler()),
    #                   ('km', kmeans(random_state=1))]
    # km_pipe = Pipeline(pipeline_order)

    # dataX_train = StandardScaler().fit_transform(dataX_train)
    km = kmeans(random_state=1)

    if k < 9999:
        km.set_params(n_clusters=k)
        km.fit(dataX_train)
        train_labels = km.labels_
        km.fit(dataX_test)
        test_labels = km.labels_
        return train_labels, test_labels  #km.transform(dataX_train), km.transform(dataX_test)


    n_clusters =  [2,3,4,5,6,7,8,9,10,15,20]
    num_k = np.shape(n_clusters)
    fit_score = np.zeros(num_k[0])
    CHI = np.zeros(num_k[0])
    DBI = np.zeros(num_k[0])

    i=0
    for k in n_clusters:
        # print(k)
        km.set_params(n_clusters=k)
        km.fit(dataX_train)
        if k<=20:
            if np.shape(dataX_train)[1] > 2:
                myplots.visualize_3d(dataX_train, km.labels_, algorithm="tsne", #"pca",
                             title="part " + str(part) + ", " + str(ds) + ", 3D TSNE Kmeans " + str(k) + " clusters",
                             filename="part " + str(part) + ", " + str(ds) + ", 3D TSNE Kmeans " + str(k) + " clusters"+timestr)
        plt1 = myplots.visualize_2d(dataX_train, km.labels_, algorithm="tsne", #"pca",
                             title="part " + str(part) + ", " + str(ds) + ", Kmeans, " + str(k) + " clusters")
        plt1.savefig("Part " + str(part) + " " + ds + " 2D TSNE viz for KMeans with " + str(k) + " clusters"+timestr + '.png')
        plt1.close()

        fit_score[i] = km.score(dataX_train)
        CHI[i] = chi(dataX_train, km.labels_)
        DBI[i] = dbi(dataX_train, km.labels_)
        homo,comp,vmeas = myplots.compute_accuracy(datay_train,km.labels_)
        print("homo,comp,vmeas for Part " + str(part) + ", " + ds + ", " + str(k) + " clusters, Kmeans: " + str(homo)+ ", "+ str(comp)+ ", "+ str(vmeas))
        i+=1

        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(dataX_train) + (k + 1) * 10])

        cluster_labels = km.fit_predict(dataX_train)
        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(dataX_train, cluster_labels)
        # print("For n_clusters =", k,
        #       "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(dataX_train, cluster_labels)

        y_lower = 10
        for j in range(k):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == j]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(j) / k)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(j))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / k)
        ax2.scatter(dataX_train[:, 0], dataX_train[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

        # Labeling the clusters
        centers = km.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')

        for j, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % j, alpha=1,
                        s=50, edgecolor='k')

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(("Part " + str(part) + " " + ds + " Silhouette analysis for KMeans clustering on sample data "
                      "with n_clusters = %d" % k),
                     fontsize=14, fontweight='bold')

        plt.savefig("Part " + str(part) + " " + ds + " Silhouette analysis for KMeans clustering on sample data "
                      "with n_clusters = " + str(k) + timestr + '.png')
        plt.close()
    # plt.show()


    # myplots.visualize_3d(dataX_train, datay_train, algorithm="tsne", #"pca",
    #                      title=str(ds) + ", part " + str(part) + "TSNE visualization")

    print(fit_score)
    print(CHI)

    # Davies Bouldin score for K means
    print(DBI)
    _, ax = plt.subplots()  # Create a new figure
    plt.plot(n_clusters, DBI, linestyle='--', marker='o', color='b')
    plt.xlabel('K')
    plt.ylabel('Davies Bouldin score')
    plt.title('Davies Bouldin score vs. K'+ " " + ds + "Part " + str(part))
    plt.savefig("Part " + str(part) + " " + ds + " Davies Bouldin score vs. K" + timestr + '.png')
    # plt.show()

    # Elbow Method for K means
    _, ax = plt.subplots()  # Create a new figure
    visualizer1 = KElbowVisualizer(km, k=(2, 30), timings=True)
    visualizer1.fit(dataX_train)  # Fit data to visualizer
    visualizer1.show(outpath="Part " + str(part) + " " + ds + " Elbow Method for K means" + timestr + '.png')
    # visualizer.show()  # Finalize and render figure
    # visualizer1.close()

    # Silhouette Score for K means
    _, ax = plt.subplots()  # Create a new figure
    visualizer2 = KElbowVisualizer(km, k=(2, 30), metric='silhouette', timings=True)
    visualizer2.fit(dataX_train)  # Fit the data to the visualizer
    visualizer2.show(outpath="Part " + str(part) + " " + ds + " Silhouette Score for K means" + timestr + '.png')
    # visualizer.show()  # Finalize and render the figure
    # visualizer.close()

    # Calinski Harabasz Score for K means
    _, ax = plt.subplots()  # Create a new figure
    visualizer3 = KElbowVisualizer(km, k=(2, 30), metric='calinski_harabasz', timings=True)
    visualizer3.fit(dataX_train)  # Fit the data to the visualizer
    visualizer3.show(outpath="Part " + str(part) + " " + ds + " Calinski Harabasz Score for K means" + timestr + '.png')
    # visualizer.show()  # Finalize and render the figure
    # visualizer.close()



    # lowest_bic = np.infty
    # bic = []
    # n_components_range = range(1, 7)
    # cv_types = ['spherical', 'tied', 'diag', 'full']
    # for cv_type in cv_types:
    #     for n_components in n_components_range:
    #
    #         km.set_params(n_clusters=n_components, covariance_type=cv_type)
    #         km.fit(dataX_train)
    #         # if k < 10: myplots.visualize_3d(dataX_train, km.labels_, algorithm="tsne",  # "pca",
    #         #                                 title="part " + str(part) + ", " + str(ds) + ", " + str(k) + " clusters")
    #         # fit_score[i] = km.score(dataX_train)
    #         # CHI[i] = chi(dataX_train, km.labels_)
    #         # DBI[i] = dbi(dataX_train, km.labels_)
    #         # i += 1
    #         bic.append(km.bic(dataX_train))
    #         if bic[-1] < lowest_bic:
    #             lowest_bic = bic[-1]
    #             best_km = km
    #
    # bic = np.array(bic)
    # color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
    #                               'darkorange'])
    # clf = best_km
    # bars = []
    #
    # # Plot the BIC scores
    # plt.figure(figsize=(8, 6))
    # spl = plt.subplot(2, 1, 1)
    # for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
    #     xpos = np.array(n_components_range) + .2 * (i - 2)
    #     bars.append(plt.bar(xpos, bic[i * len(n_components_range):
    #                                   (i + 1) * len(n_components_range)],
    #                         width=.2, color=color))
    # plt.xticks(n_components_range)
    # plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
    # plt.title('BIC score per model')
    # xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 + \
    #        .2 * np.floor(bic.argmin() / len(n_components_range))
    # plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
    # spl.set_xlabel('Number of components')
    # spl.legend([b[0] for b in bars], cv_types)
    #
    # # Plot the winner
    # splot = plt.subplot(2, 1, 2)
    # Y_ = clf.predict(dataX_train)
    # for i, (mean, cov, color) in enumerate(zip(clf.means_, clf.covariances_,
    #                                            color_iter)):
    #     v, w = linalg.eigh(cov)
    #     if not np.any(Y_ == i):
    #         continue
    #     plt.scatter(dataX_train[Y_ == i, 0], dataX_train[Y_ == i, 1], .8, color=color)
    #
    #     # Plot an ellipse to show the Gaussian component
    #     angle = np.arctan2(w[0][1], w[0][0])
    #     angle = 180. * angle / np.pi  # convert to degrees
    #     v = 2. * np.sqrt(2.) * np.sqrt(v)
    #     ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
    #     ell.set_clip_box(splot.bbox)
    #     ell.set_alpha(.5)
    #     splot.add_artist(ell)
    #
    # plt.xticks(())
    # plt.yticks(())
    # plt.title(f'Selected GMM: {best_km.covariance_type} model, '
    #           f'{best_km.n_components} components')
    # plt.subplots_adjust(hspace=.35, bottom=.02)
    # # plt.show()

    # Inertia elbow method
    # Silhouette
    # Calinski-Harabasz Index
    # Davies-Bouldin Index





# In theory if you make enough clusters, every single datapoint is in its own cluster and is perfectly well identified.
# If you're using scikit there are four ways to make non-ground truth decisions:
# Inertia elbow method
# Silhouette
# Calinski-Harabasz Index
# Davies-Bouldin Index
# Each method has different strength/weaknesses and use cases. And you don't need to pick just one to make decisions
# with, in fact you could use multiple. It's worth comparing multiple.
# Further you should exercise some just judgement on the dataset, I don't know if there's a common rule of thumb,
# but the number of clusters probably should not exceed 5% the size of the dataset
#
# Michael Driscoll:classical_building:  8 days ago
# And even 5% seems high
#
# Taneem  8 days ago
# @Michael Driscoll Does inertia elbow applies to anything beyond k-means? I have not made if further than k-means yet
# but so far it seems elbow is pretty much focused on k-means selection of k.
#
# Michael Driscoll:classical_building:  8 days ago
# It apples to EM as well