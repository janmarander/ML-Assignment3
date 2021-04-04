# Use Gaussian for Randomized projection

from sklearn import decomposition
from sklearn.random_projection import GaussianRandomProjection
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
from itertools import product



from scipy import linalg
import matplotlib as mpl

from sklearn import mixture
import A1_Decision_Trees as dt

# https://learning.oreilly.com/library/view/hands-on-unsupervised-learning/9781492035633/ch04.html
# def loss_Reconstruction_Error(originalDF, reducedDF):
#     loss = np.sum((np.array(originalDF)-np.array(reducedDF))**2, axis=1)
#     loss = pd.Series(data=loss,index=originalDF.index)
#     loss = (loss-np.min(loss))/(np.max(loss)-np.min(loss))
#     return loss





def grp(dataX_train, datay_train, dataX_test, datay_test, timestr, n=9999, part='1', ds = 'Set1'):
    print("Part "+str(part))
    print(ds)
    print("Randomized Projections (Gaussian)")

    if n < 9999:
        num_repeats = 20
        best_results = dataX_train
        best_test = dataX_test
        error = 99999
        for r in range(num_repeats):
            rp = GaussianRandomProjection(n_components=n,random_state=r)
            rp.set_params(n_components=n)
            transformed_data = rp.fit_transform(dataX_train)
            transformed_test = rp.transform(dataX_test)
            print(rp.components_.T)
            inverse_data = np.linalg.pinv(rp.components_.T)

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



    i = 0
    num_repeats = 20
    rerr = np.zeros((num_repeats,num_c[0]))
    print(np.shape(rerr))
    best_results = dataX_train
    best_test = dataX_test
    error = 99999
    for n in comps:
        for r in range(num_repeats):
            rp = GaussianRandomProjection(random_state=r)
            rp.set_params(n_components=n)
            transformed_data = rp.fit_transform(dataX_train)
            transformed_test = rp.transform(dataX_test)
            inverse_data = np.linalg.pinv(rp.components_.T)
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
    plt.title(str(ds) +'RP Gaussian: RMSE Reconstruction Error vs. num components')

    plt.savefig("Part " + str(part) + " " + ds + " Reconstruction Error for RP Gaussian" + timestr + '.png')
    # plt.show()
    plt.close()


    # From Piazza Office Hours thread:
    # "Something like:
    # transformed_data = (...).fit_transform(X)
    # inverse_data = np.linalg.pinv((...).components_.T)
    # reconstructed_data = transformed_data.dot(inverse_data)
    # ... if you're using SKLearn. Then minimize error."


#So since we're supposed to run Random Projection multiple times, once we've settled on some best n_components, is
# there like a way to like average/combine the transformations caused by each different random projection at that
# n_components? Or are you guys just picking one random run at the best n_components and using that run's transformed
# data for everything else? (edited)

# The latter for me

# I’m not sure if its the best way to do it, but i’m doing 20 runs and then plotting the average of the runs and the
# standard deviation of the runs

# average reconstruction error

#Thread
# cs7641
#
# Tapan Mar 30th at 12:02 AM
# So how is everyone picking number of components in random projection. My reconstruction error just keeps going down with number of components and I still haven't figured out what's the best way
#
# that's what a lot of people have experienced. I'm not so sure on the right approach lol
#
# So far I have it thresholded like we discussed the other night
#
# but didn't know if somebody had come up with a better idea
#
# I haven't seen one. I think this might be a good thing for OH, though
#
# I guess I'll wait till the OH lol
#
# i've posted so many times in that thread lol
#
# We need upvote buttons to move questions up or down :rolling_on_the_floor_laughing:
#
# I would go with a combination of picking a threshold and elbow method. once the gains from increasing the number of components diminishes that it an area you can look at for choosing
#
# I used reconstruction error to compute thresholds. PCA of 95 variance gave RMSE ~300 in my case. So I've used 300 RMSE as threshold for ICA and RP. I still have to verify if this threshold is yielding better clusters though.







