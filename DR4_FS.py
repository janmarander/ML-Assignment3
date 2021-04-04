from sklearn import decomposition
from sklearn.random_projection import SparseRandomProjection
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
import scipy.sparse as sps
import A1_Decision_Trees as dt

# https://learning.oreilly.com/library/view/hands-on-unsupervised-learning/9781492035633/ch04.html
# def loss_Reconstruction_Error(originalDF, reducedDF):
#     loss = np.sum((np.array(originalDF)-np.array(reducedDF))**2, axis=1)
#     loss = pd.Series(data=loss,index=originalDF.index)
#     loss = (loss-np.min(loss))/(np.max(loss)-np.min(loss))
#     return loss





def srp(dataX_train, datay_train, dataX_test, datay_test, timestr, n=9999, part='1', ds = 'Set1'):
    print("Part "+str(part))
    print(ds)
    print("Randomized Projections (Sparse)")
    print(n)

    if n < 9999:
        num_repeats = 20
        best_results = dataX_train
        best_test = dataX_test
        error = 99999
        for r in range(num_repeats):
            rp = SparseRandomProjection(n_components=n,random_state=r)
            transformed_data = rp.fit_transform(dataX_train)
            transformed_test = rp.transform(dataX_test)
            coms = rp.components_
            if sps.issparse(coms): coms = coms.todense()
            inverse_data = np.linalg.pinv(coms.T)
            reconstructed_data = transformed_data.dot(inverse_data)
            rmse = np.square(dataX_train-reconstructed_data)
            rmse = (np.sqrt(np.nanmean(rmse))) /np.mean(abs(dataX_train))
            #print(rmse)
            if rmse < error:
                error = rmse
                best_results = transformed_data
                best_test = transformed_test
        return best_results, best_test

    comps = [2, 3, 4, 5, 6, 7, 8]
    num_c = np.shape(comps)
    num_repeats = 20
    rerr = np.zeros((num_repeats, num_c[0]))
    best_results = dataX_train
    best_test = dataX_test
    error = 99999
    i = 0
    for n in comps:
        for r in range(num_repeats):
            rp = SparseRandomProjection(random_state=r)
            rp.set_params(n_components=n)
            transformed_data = rp.fit_transform(dataX_train)
            transformed_test = rp.transform(dataX_test)
            inverse_data = np.linalg.pinv((rp.components_).todense().T)
            reconstructed_data = transformed_data.dot(inverse_data)
            rmse = np.square(dataX_train-reconstructed_data)
            #print(rmse)
            rerr[r,i] = ((np.nanmean(rmse))) #(np.sqrt(np.nanmean(rmse))) /np.mean(abs(dataX_train)) #np.nanmean(rmse)
            rmse = (np.sqrt(np.nanmean(rmse))) / np.mean(abs(dataX_train))
            if rmse < error:
                error = rmse
                best_results = transformed_data
                best_test = transformed_test
        dt.DT(best_results, best_test, datay_train, datay_test, timestr, part=part, ds=ds)
        i += 1

    re_stdev = np.std(rerr, axis=0)
    re_mean = np.mean(rerr, axis=0)

    _, ax = plt.subplots()  # Create a new figure
    plt.plot(comps, re_mean, linestyle='--', marker='o', color='b')
    plt.fill_between(comps, re_mean-re_stdev, re_mean+re_stdev, color='b', alpha=0.5)
    plt.xlabel('K')
    plt.ylabel('RMSE Reconstruction Error')
    plt.title(str(ds) +' RP Sparse: RMSE Reconstruction Error vs. num components')
    plt.savefig("Part " + str(part) + " " + ds + " Reconstruction Error for RP Sparse" + timestr + '.png')
    # plt.show()
    plt.close()



# Andrew Parmar  10:39 PM
# What are you guys using for the 4th features selection algorithm? I’ve been considering LDA or FA (Factor Analysis)
#
#
# I've initially considered Feature Agglomeration but decided to use LDA because it was also discussed in the
# lectures, which means I'll probably be able to explain it better.


# Taneem Mar 26th at 11:42 PM
# For RandomProjection do we use Sparse or Gaussian or do we care?
#
#
#
#
# 3 replies
#
# Tapan  1 day ago
# I dont think it matters but in one of the OH, somebody had the Gaussian listed and TAs said, it's good.
#
# Taneem  1 day ago
# Okay!
#
# K  1 day ago
# You can use the other as the 4th method
