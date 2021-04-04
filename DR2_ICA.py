# 2.	Apply the dimensionality reduction algorithms to the two datasets and describe what you see.

from sklearn import decomposition
from sklearn.decomposition import FastICA
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

def ica(dataX_train, datay_train, dataX_test, datay_test, timestr, n=9999, part='1', ds = 'Set1'):
    print("Part "+str(part))
    print(ds)
    print("Independent Component Analysis")

    if n < 9999:
        ica = FastICA(n_components=n, random_state=1, max_iter=25000, tol=.05)
        return ica.fit_transform(dataX_train), ica.transform(dataX_test)

    comps = [1,2,3,4,5,6,7,8]
    # raise
    # %% data for 1
    num_c = np.shape(comps)
    kurtosis = np.zeros(num_c[0])
    rerr = np.zeros(num_c[0])

    ica = FastICA(random_state=1, max_iter=15000, tol=.05)
    i = 0
    for n in comps:
        print(n)
        ica.set_params(n_components=n, max_iter=25000, tol=.05)
        res = ica.fit_transform(dataX_train)
        res = pd.DataFrame(res)
        res_kurt = res.kurt(axis=0)
        kurtosis[i] = res_kurt.abs().mean()
        transformed_data = res
        inverse_data = np.linalg.pinv(ica.components_.T)
        reconstructed_data = transformed_data.dot(inverse_data)
        rmse = np.square(dataX_train - reconstructed_data)
        rerr[i] = np.nanmean(rmse) #(np.sqrt(np.nanmean(rmse))) /np.mean(abs(dataX_train)) # np.nanmean(rmse) #
        dt.DT(ica.transform(dataX_train), ica.transform(dataX_test), datay_train, datay_test, timestr, part=part, ds=ds)
        i += 1

        # W = projections.components_
        # p = pinv(projections.components_)
        # reconstructed = ((p @ W) @ (X.T)).T  # Unproject projected data
        # errors = np.square(X - reconstructed)

    _, ax = plt.subplots()  # Create a new figure
    plt.plot(comps, rerr, linestyle='--', marker='o', color='b')
    plt.xlabel('K')
    plt.ylabel('RMSE Reconstruction Error')
    plt.title(str(ds) + ' ICA RMSE Reconstruction Error vs. num components')

    # plt.show()
    # "mean ***kurtosis*** for ICA"
    plt.plot(comps,kurtosis)
    plt.xlabel('number of components')
    plt.ylabel('kurtosis')
    plt.title(str(ds) + " ICA Kurtosis")
    plt.savefig("Part " + str(part) + " " + ds + " ICA Kurtosis" + timestr + '.png')
    # plt.show()
    plt.close()

#you look at kurtosis in ica to find it vs variance in pca

# toheeb Today at 1:25 AM
# any resources to help understand how to interpret kurtosis in ICA? (edited)
# 7 replies
#
# Mark  11 hours ago
# interpret how? Kurtosis is helpful to think of in relation to a normal distribution (with a kurtosis of ~3). Smaller
# values spread the curve out more so values are more likely to fall in the tails. Larger values make it much taller
# and spikier, meaning values are much less likely to fall in the tails. The reason why we maximize kurtosis is because
# we want to maximize independence, and to do that we want to minimize the likelihood that a datapoint falls outside of
# the cluster we've assigned it. So high avg kurtosis = more independent clusters = more likely that our label is
# correct (edited)
#
# toheeb  11 hours ago
# this is very helpful mark!
#
# toheeb  11 hours ago
# I guess I meant understand kurtosis with respects to ICA which your last few sentences addressed, thank you
#
# Kunal Sethi:masterball:  10 hours ago
# Great explanation @Mark, one thing to note - if you are using pandas to compute kurtosis, DataFrame.kurt uses
# Fisher's definition of kurtosis where a normal distribution  == 0.0 instead of 3.

#
# sai  10 hours ago
# To compute the kurtosis, do we run this on the fit_transform output? That's what was mentioned in OH. Wanted to
# clarify if this is how everyone is doing it or do we run it on the ica.components_? This is confusing to me.
#
# Neha Gupta  8 hours ago
# I interpret it as larger value of kurtosis means more spread out curve in the tails which means our data is more non
# Gaussian and for ICA , that is the key.
#
# Neha Gupta  8 hours ago
# If kurtosis is small , it means my data is more Gaussian as most of the data reside in the bell curve not on the
# tails side.