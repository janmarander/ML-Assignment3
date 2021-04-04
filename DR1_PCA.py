# 2.	Apply the dimensionality reduction algorithms to the two datasets and describe what you see.

from sklearn import decomposition
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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

import A1_Decision_Trees as dt


def pca(dataX_train, datay_train, dataX_test, datay_test, timestr, n=9999, part='1', ds = 'Set1'):
    print("Part "+str(part))
    print(ds)
    print("principle component analysis")

    np.random.seed(42)

    n_components = [1,2,3,4,5,6,7,8,9]

    # for n in n_components:
    #     pca = PCA(n_components = n,random_state=1)
    #     pca.fit(dataX_train)

    if n < 9999:
        pca = PCA(n_components=n, random_state=1)
        pca.fit(dataX_train)
        X_pca = pca.transform(dataX_train)
        print("original shape:   ", dataX_train.shape)
        print("transformed shape:", X_pca.shape)
        return pca.transform(dataX_train), pca.transform(dataX_test)

    pca = PCA().fit(dataX_train)
    _, ax = plt.subplots()  # Create a new figure
    evr = np.cumsum(pca.explained_variance_ratio_)
    plt.plot(n_components,evr)
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.title(str(ds) + " PCA Cumulative Explained Variance")
    plt.savefig("Part " + str(part) + " " + ds + " PCA Cumulative Explained Variance" + timestr + '.png')
    # plt.show()
    plt.close()
#https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html
    print(pca.explained_variance_ratio_)
    for n_comp in n_components:
        print("Num Comp: ", n_comp)
        pca = PCA(n_components=n_comp, random_state=1)
        pca.fit(dataX_train)
        X_pca = pca.transform(dataX_train)
        print("original shape:   ", dataX_train.shape)
        print("transformed shape:", X_pca.shape)
        if n_comp>0: dt.DT(pca.transform(dataX_train), pca.transform(dataX_test), datay_train, datay_test, timestr, part=part, ds=ds)

#you look at kurtosis in ica to find it vs variance in pca

# How are people selecting the number of components to take in PCA? (haven't explored the other reduction methods yet)
# It seems there's an obvious cliff of which components are providing value. Thinking i'll take this. (edited)
#
# You just need to determine an acceptable variance percentile based on your domain knowledge. A common value is 95%
# of variance with X number of components
#
# Taneem  23 hours ago
# But you dont have to stick to 95% (you can say something like based on my domain knowledge on the data, 80% variance
# is good enough).
#
#
# I have seen a few references to using kurtosis for ICA as well.
# Still investigating what that actually means. :laughing:
#
# pandas dataframe has a kurtosis built in which under the hood uses scipy kurtosis
#
# I ended up using scipy one directly since it allows you to play around with various measures
#
# are you calculating kurtosis on every component?
#
# yes, by component. Then average the component kurtosises (kurtoses?)
#
# I may be off in the woods here, but to confirm my understanding. The idea is to find n_components such that the
# kurtosis of each output feature is as close to 0 as possible? We want each individual output feature in the
# transformation to be normally distributed?
#
# Jessi Beck  21 hours ago
# I think you want the higher kurtosis one. The notes i took on yesterday's OH say "pick the most non Gaussian
# solution (use kurtosis - larger kurtosis than normal gaussian (in normal magnitude), is more non-Gaussian)"
#
# Taneem  21 hours ago
# right the max kurt … however, depending on your dataset you may not see an obvious fall off
#
# Taneem  21 hours ago
# you will know what I mean when you plot it .
#
# Steven Spohrer  21 hours ago
# What is the intuition between wanting the most non gaussian solution?
#
# Taneem  21 hours ago
# “independence”
#
# Steven Spohrer  21 hours ago
# Maybe I am misunderstanding the approach..
# Why does having a non-normal distribution of each component mean they are independent of the other components?
#
# Taneem  21 hours ago
# oh man … you have to dive into the central limit theorem and stuff to make full sense of it

#
#

#How do we compare clusters before and after dimensionality reduction?
# Devin Hobby  3 days ago
# I was planning to do a performance comparison which has a couple factors:
# Accuracy/Error
# Number of Clusters
# Execution Time
#
# ML  3 days ago
# How would you assess accuracy?
# :thisalpha:
# 1
#
#
# Steven Spohrer  21 hours ago
# Are you using the same number of clusters as the clustering before reduction?
#
# jsiddon  21 hours ago
# I am figuring out optimal number of clusters, then recording my metrics, then running DR, then again recording
# metrics for optimal clusters
#
# Devin Hobby  20 hours ago
# There are a couple different methods, but I would just pick the one or two that you understand the most so you can
# write about them. Just make sure to keep the analysis consistent before and after FT and FS. Here are some of the ones
# I used for kneads for example:
# kmeans.score(x)
# normalized_mutual_info_score(y, labels)
# silhouette_score(x, labels)
# davies_bouldin_score(x, predictions)
# v_measure_score(y, predictions)
# accuracy_score(y, predictions)
