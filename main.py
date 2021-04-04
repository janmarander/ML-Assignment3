# Assignment 3
#
# You are to implement (or find the code for) six algorithms. The first two are clustering algorithms:
# •	k-means clustering
# •	Expectation Maximization
# You can choose your own measures of distance/similarity. Naturally, you'll have to justify your choices, but you're
# practiced at that sort of thing by now.

# The last four algorithms are dimensionality reduction algorithms:
# •	PCA
# •	ICA
# •	Randomized Projections
# •	Any other feature selection algorithm you desire

# You are to run a number of experiments. Come up with at least two datasets. If you'd like (and it makes a lot of
# sense in this case) you can use the ones you used in the first assignment.
# 1.	Run the clustering algorithms on the datasets and describe what you see.
# 2.	Apply the dimensionality reduction algorithms to the two datasets and describe what you see.
# 3.	Reproduce your clustering experiments, but on the data after you've run dimensionality reduction on it. Yes,
# that’s 16 combinations of datasets, dimensionality reduction, and clustering method. You should look at all of them,
# but focus on the more interesting findings in your report.
# 4.	Apply the dimensionality reduction algorithms to one of your datasets from assignment #1 (if you've reused the
# datasets from assignment #1 to do experiments 1-3 above then you've already done this) and rerun your neural network
# learner on the newly projected data.
# 5.	Apply the clustering algorithms to the same dataset to which you just applied the dimensionality reduction
# algorithms (you've probably already done this), treating the clusters as if they were new features. In other words,
# treat the clustering algorithms as if they were dimensionality reduction algorithms. Again, rerun your neural network
# learner on the newly projected data.

# In part 4 & 5, do we need to run the neural network with different number of clusters and components? Or is it
# sufficient to run only with a number of clusters / components we selected as best in part 1 & 2 and compare with
# the base case? (A) Run with the optimal K clusters and N components you found.


import sklearn as sk
import sklearn.model_selection
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import make_plots as myplots
import plotly
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import make_plots
#from make_plots import plot_learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

import C1_kmeans as km
import C2_EM as em
import DR1_PCA as pca
import DR2_ICA as ica
import DR3_RP as rp
import DR4_FS as fs
import P4_NN as nn

import sys
import time

if __name__ == '__main__':
    stdoutOrigin = sys.stdout
    print(stdoutOrigin)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    sys.stdout = open("log"+timestr+".txt", "w")
    print(timestr)

    print('ML Assignment 3')
    # read in data:
    set1 = pd.read_hdf('alldata.hdf', 'set1')
    ny = set1.shape[1] - 1
    set1X = set1.drop(ny, 1).copy().values
    set1Y = set1[ny].copy().values
    # print(set1)
    # print(set1Y)

    set2 = pd.read_hdf('alldata.hdf', 'set2b')
    ny = set2.shape[1] - 1
    set2X = set2.drop(ny, 1).copy().values
    set2Y = set2[ny].copy().values
    # print(set2)
    # print(set2Y)

    # split into train and test data
    set1X_train, set1X_test, set1y_train, set1y_test = sk.model_selection.train_test_split(set1X, set1Y, test_size=0.3,
                                                                                           random_state=42,stratify=set1Y)
    set2X_train, set2X_test, set2y_train, set2y_test = sk.model_selection.train_test_split(set2X, set2Y, test_size=0.3,
                                                                                           random_state=42,stratify=set2Y)

    set1X_train = StandardScaler().fit_transform(set1X_train)
    set2X_train = StandardScaler().fit_transform(set2X_train)



    timestr = "-" + time.strftime("%Y%m%d-%H%M%S")

    # plt = myplots.visualize_2d(set1X_train, set1y_train, algorithm="tsne",  # "pca",
    #                            title="TSNE 2D Original Data and Labels, Set1")
    # plt.savefig(
    #     "TSNE 2D Original Data and Labels, Set1" + timestr + '.png')
    # plt.close()
    #
    # myplots.visualize_3d(set1X_train, set1y_train, algorithm="tsne",  # "pca",
    #                      title="TSNE 3D Original Data and Labels, Set1",
    #                      filename="TSNE 3D Original Data and Labels, Set1" + timestr)
    #
    # plt = myplots.visualize_2d(set2X_train, set2y_train, algorithm="tsne",  # "pca",
    #                            title="TSNE 2D Original Data and Labels, Set2")
    # plt.savefig(
    #     "TSNE 2D Original Data and Labels, Set2" + timestr + '.png')
    # plt.close()
    #
    # myplots.visualize_3d(set2X_train, set2y_train, algorithm="tsne",  # "pca",
    #                      title="TSNE 3D Original Data and Labels, Set2",
    #                      filename="TSNE 3D Original Data and Labels, Set2" + timestr)
    #


########################################################################################################################
    # sys.stdout.close()
    # sys.stdout = stdoutOrigin
    # stdoutOrigin = sys.stdout
    # sys.stdout = open("Part1"+timestr+".txt", "w")
    #
    # plt.close('all')

    # ## 1.	Run the clustering algorithms on the datasets and describe what you see.
    # ## •	k-means clustering
    # km.km(set1X_train, set1y_train, set1X_test, set1y_test, timestr, part = "1", ds = 'Set1')
    # km.km(set2X_train, set2y_train, set2X_test, set2y_test, timestr, part = "1", ds = 'Set2')
    # # •	Expectation Maximization
    # em.em(set1X_train, set1y_train, set1X_test, set1y_test, timestr, part = "1", ds = 'Set1')
    # em.em(set2X_train, set2y_train, set2X_test, set2y_test, timestr, part = "1", ds = 'Set2')
    #


########################################################################################################################
    sys.stdout.close()
    sys.stdout = stdoutOrigin
    stdoutOrigin = sys.stdout
    sys.stdout = open("Part2"+timestr+".txt", "w")

    plt.close('all')

    # # 2.	Apply the dimensionality reduction algorithms to the two datasets and describe what you see.
    # # •	PCA
    pca.pca(set1X_train, set1y_train, set1X_test, set1y_test, timestr, n=9999, part='2', ds='Set1')
    pca.pca(set2X_train, set2y_train, set2X_test, set2y_test, timestr, n=9999, part='2', ds='Set2')
    print("-------------------------------")
    #•	ICA
    ica.ica(set1X_train, set1y_train, set1X_test, set1y_test, timestr, n=9999, part='2', ds='Set1')
    ica.ica(set2X_train, set2y_train, set2X_test, set2y_test, timestr, n=9999, part='2', ds='Set2')
    # print("-------------------------------")
    # # •	Randomized Projections
    rp.grp(set1X_train, set1y_train, set1X_test, set1y_test, timestr, n=9999, part='2', ds='Set1')
    rp.grp(set2X_train, set2y_train, set2X_test, set2y_test, timestr, n=9999, part='2', ds='Set2')
    # print("-------------------------------")
    # •	Any other feature selection algorithm you desire
    fs.srp(set1X_train, set1y_train, set1X_test, set1y_test, timestr, n=9999, part='2', ds='Set1')
    fs.srp(set2X_train, set2y_train, set2X_test, set2y_test, timestr, n=9999, part='2', ds='Set2')




########################################################################################################################
    # sys.stdout.close()
    # sys.stdout = stdoutOrigin
    # stdoutOrigin = sys.stdout
    # sys.stdout = open("Part3"+timestr+".txt", "w")
    #
    # plt.close('all')

    # 3.	Reproduce your clustering experiments, but on the data after you've run dimensionality reduction on it. Yes,
    # that’s 16 combinations of datasets, dimensionality reduction, and clustering method. You should look at all of them,
    # but focus on the more interesting findings in your report.

    # # CALL CLUSTERING FUNCTION FROM (1) USING OUTPUT FROM (2)
    # # •	PCA
    # pca_res1, pca_test1 = pca.pca(set1X_train, set1y_train, set1X_test, set1y_test, timestr, n=7, part='3', ds='Set1')
    #
    # plt = myplots.visualize_2d(pca_res1, set1y_train, algorithm="tsne",  # "pca",
    #                            title="TSNE 2D PCA-Reducted Data with Original Labels, Set1")
    # plt.savefig(
    #     "Part 3 TSNE 2D PCA-Reducted Data with Original Labels, Set1" + timestr + '.png')
    # plt.close()
    #
    # myplots.visualize_3d(pca_res1, set1y_train, algorithm="tsne",  # "pca",
    #                      title="TSNE 3D Reduced Data and Labels, Set1",
    #                      filename="Part 3 TSNE 3D PCA-Reducted Data with Original Labels, Set1" + timestr)
    #
    # pca_res2, pca_test2 = pca.pca(set2X_train, set2y_train, set2X_test, set2y_test, timestr, n=4, part='3', ds='Set2')
    # plt = myplots.visualize_2d(pca_res2, set1y_train, algorithm="tsne",  # "pca",
    #                            title="TSNE 2D PCA-Reducted Data with Original Labels, Set2")
    # plt.savefig(
    #     "Part 3 TSNE 2D PCA-Reducted Data with Original Labels, Set2" + timestr + '.png')
    # plt.close()
    #
    # myplots.visualize_3d(pca_res2, set2y_train, algorithm="tsne",  # "pca",
    #                      title="TSNE 3D Reduced Data and Labels, Set2",
    #                      filename="Part 3 TSNE 3D PCA-Reducted Data with Original Labels, Set2" + timestr)
    #
    # km.km(pca_res1, set1y_train, set1X_test, set1y_test, timestr, part = "3", ds = 'Set1_PCA')
    # km.km(pca_res2, set1y_train, set1X_test, set1y_test, timestr, part="3", ds='Set2_PCA')
    # em.em(pca_res1, set1y_train, set1X_test, set1y_test, timestr, part = "3", ds = 'Set1_PCA')
    # em.em(pca_res2, set1y_train, set1X_test, set1y_test, timestr, part="3", ds='Set2_PCA')
    #
    # # •	ICA
    # ica_res1, ica_test1 = ica.ica(set1X_train, set1y_train, set1X_test, set1y_test, timestr, n=2, part='3', ds='Set1')
    # ica_res2, ica_test2 = ica.ica(set2X_train, set2y_train, set2X_test, set2y_test, timestr, n=5, part='3', ds='Set2')
    # km.km(ica_res1, set1y_train, set1X_test, set1y_test, timestr, part = "3", ds = 'Set1_ICA')
    # km.km(ica_res2, set1y_train, set1X_test, set1y_test, timestr, part="3", ds='Set2_ICA')
    # em.em(ica_res1, set1y_train, set1X_test, set1y_test, timestr, part = "3", ds = 'Set1_ICA')
    # em.em(ica_res2, set1y_train, set1X_test, set1y_test, timestr, part="3", ds='Set2_ICA')

    # plt = myplots.visualize_2d(ica_res1, set1y_train, algorithm="tsne",  # "pca",
    #                            title="TSNE 2D ICA-Reducted Data with Original Labels, Set1")
    # plt.savefig(
    #     "Part 3 TSNE 2D ICA-Reducted Data with Original Labels, Set1" + timestr + '.png')
    # plt.close()
    #
    # if np.shape(ica_res1)[1] > 2: myplots.visualize_3d(ica_res1, set1y_train, algorithm="tsne",  # "pca",
    #                      title="TSNE 3D Original Data and Labels, Set1",
    #                      filename="Part 3 TSNE 3D ICA-Reducted Data with Original Labels, Set1" + timestr)
    #
    # plt = myplots.visualize_2d(ica_res2, set2y_train, algorithm="tsne",  # "pca",
    #                            title="TSNE 2D ICA-Reducted Data with Original Labels, Set2")
    # plt.savefig(
    #     "Part 3 TSNE 2D ICA-Reducted Data with Original Labels, Set2" + timestr + '.png')
    # plt.close()
    #
    # if np.shape(ica_res2)[1] > 2: myplots.visualize_3d(ica_res2, set2y_train, algorithm="tsne",  # "pca",
    #                      title="TSNE 3D Original Data and Labels, Set2",
    #                      filename="Part 3 TSNE 3D ICA-Reducted Data with Original Labels, Set2" + timestr)
    #
    # plt.close('all')
    # # •	Randomized Projections (Gaussian)
    # rpg_res1, rpg_test1 = rp.grp(set1X_train, set1y_train, set1X_test, set1y_test, timestr, n=4, part='3', ds='Set1')
    # rpg_res2, rpg_test2 = rp.grp(set2X_train, set2y_train, set2X_test, set2y_test, timestr, n=6, part='3', ds='Set2')
    # km.km(rpg_res1, set1y_train, set1X_test, set1y_test, timestr, part = "3", ds = 'Set1_RPG')
    # km.km(rpg_res2, set1y_train, set1X_test, set1y_test, timestr, part="3", ds='Set2_RPG')
    # em.em(rpg_res1, set1y_train, set1X_test, set1y_test, timestr, part = "3", ds = 'Set1_RPG')
    # em.em(rpg_res2, set1y_train, set1X_test, set1y_test, timestr, part="3", ds='Set2_RPG')
    #
    # plt = myplots.visualize_2d(rpg_res1, set1y_train, algorithm="tsne",  # "pca",
    #                            title="TSNE 2D RP(G)-Reducted Data with Original Labels, Set1")
    # plt.savefig(
    #     "Part 3 TSNE 2D RP(G)-Reducted Data with Original Labels, Set1" + timestr + '.png')
    # plt.close()
    #
    # if np.shape(rpg_res1)[1] > 2: myplots.visualize_3d(rpg_res1, set1y_train, algorithm="tsne",  # "pca",
    #                      title="TSNE 3D Reduced Data and Labels, Set1",
    #                      filename="Part 3 TSNE 3D RP(G)-Reducted Data with Original Labels, Set1" + timestr)
    #
    # plt = myplots.visualize_2d(rpg_res2, set2y_train, algorithm="tsne",  # "pca",
    #                            title="TSNE 2D RP(G)-Reducted Data with Original Labels, Set2")
    # plt.savefig(
    #     "Part 3 TSNE 2D RP(G)-Reducted Data with Original Labels, Set2" + timestr + '.png')
    # plt.close()
    #
    # if np.shape(rpg_res2)[1] > 2: myplots.visualize_3d(rpg_res2, set2y_train, algorithm="tsne",  # "pca",
    #                      title="TSNE 3D Reduced Data and Labels, Set2",
    #                      filename="Part 3 TSNE 3D RP(G)-Reducted Data with Original Labels, Set2" + timestr)
    #
    # # •	Any other feature selection algorithm you desire (Sparse Randomized projections)
    # rps_res1, rps_test1 = fs.srp(set1X_train, set1y_train, set1X_test, set1y_test, timestr, n=4, part='3', ds='Set1')
    # rps_res2, rps_test2 = fs.srp(set2X_train, set2y_train, set2X_test, set2y_test, timestr, n=6, part='3', ds='Set2')
    # km.km(rps_res1, set1y_train, set1X_test, set1y_test, timestr, part = "3", ds = 'Set1_RPS')
    # km.km(rps_res2, set1y_train, set1X_test, set1y_test, timestr, part="3", ds='Set2_RPS')
    # em.em(rps_res1, set1y_train, set1X_test, set1y_test, timestr, part = "3", ds = 'Set1_RPS')
    # em.em(rps_res2, set1y_train, set1X_test, set1y_test, timestr, part="3", ds='Set2_RPS')
    #
    # plt = myplots.visualize_2d(rps_res1, set1y_train, algorithm="tsne",  # "pca",
    #                            title="TSNE 2D RP(S)-Reducted Data with Original Labels, Set1")
    # plt.savefig(
    #     "Part 3 TSNE 2D RP(S)-Reducted Data with Original Labels, Set1" + timestr + '.png')
    # plt.close()
    #
    # if np.shape(rps_res1)[1] > 2: myplots.visualize_3d(rps_res1, set1y_train, algorithm="tsne",  # "pca",
    #                      title="TSNE 3D Reduced Data and Labels, Set1",
    #                      filename="Part 3 TSNE 3D RP(S)-Reducted Data with Original Labels, Set1" + timestr)
    #
    # plt = myplots.visualize_2d(rps_res2, set2y_train, algorithm="tsne",  # "pca",
    #                            title="TSNE 2D RP(S)-Reducted Data with Original Labels, Set2")
    # plt.savefig(
    #     "Part 3 TSNE 2D RP(S)-Reducted Data with Original Labels, Set2" + timestr + '.png')
    # plt.close()
    #
    # if np.shape(rps_res2)[1] > 2: myplots.visualize_3d(rps_res2, set2y_train, algorithm="tsne",  # "pca",
    #                      title="TSNE 3D Reduced Data and Labels, Set2",
    #                      filename="Part 3 TSNE 3D RP(S)-Reducted Data with Original Labels, Set2" + timestr)
    #
    #         # --Have a question on Part 3. Say in Part 1 we determine our optimal number of clusters are 5. While doing Part 3
    #         # do we need to redetermine the optimal amount of clusters using the reduced dimensional data in part 2, or is
    #         # it more sore just using the reduced data for the optimal clusters found in part 1
    #         # --I’m not there yet (big rip) but I think the point is to see how your clusters are changing after dimensionality
    #         # reduction. So I imagine you need to find the optimal # again


    # sys.stdout.close()
    # sys.stdout = stdoutOrigin
    # stdoutOrigin = sys.stdout
    # sys.stdout = open("Part4"+timestr+".txt", "w")
    #
    # plt.close('all')

    ########################################################################################################################
    # 4. Apply the dimensionality reduction algorithms to one of your datasets from assignment #1 (if you've reused the
    # datasets from assignment #1 to do experiments 1-3 above then you've already done this) and rerun your neural network
    # learner on the newly projected data.

    # ## RUN NN LEARNER ON OUTPUT FROM (2)
    # ## #NN(set1X_train, set1X_test, set1y_train, set1y_test,part='4', ds='Set1')
    #
    # #re-run tuned NN from Assignment 1 to get stats:
    # nn.NN(set2X_train, set2X_test, set2y_train, set2y_test, timestr, part='5', ds='Set2_original', final=True,
    #      hidden_layer_sizes=(50,50,50), alpha=0.05)
    #
    # #
    # # pca_res2, pca_test2
    # pca_res2, pca_test2 = pca.pca(set2X_train, set2y_train, set2X_test, set2y_test, timestr, n=4, part='3', ds='Set2')
    # nn.NN(pca_res2, pca_test2, set2y_train, set2y_test, timestr, part='4', ds='Set2_PCA')
    # nn.NN(pca_res2, pca_test2, set2y_train, set2y_test, timestr, part='4', ds='Set2_PCA', final=True,hidden_layer_sizes=(20,20,20), alpha=0.1)
    #
    # # ica_res2, ica_test2
    # ica_res2, ica_test2 = ica.ica(set2X_train, set2y_train, set2X_test, set2y_test, timestr, n=5, part='3', ds='Set2')
    # nn.NN(ica_res2, ica_test2, set2y_train, set2y_test, timestr, part='4', ds='Set2_ICA')
    # nn.NN(ica_res2, ica_test2, set2y_train, set2y_test, timestr, part='4', ds='Set2_ICA', final=True,hidden_layer_sizes=(50,100), alpha=0.1)
    #
    # # rpg_res2, rpg_test2
    # rpg_res2, rpg_test2 = rp.grp(set2X_train, set2y_train, set2X_test, set2y_test, timestr, n=6, part='3', ds='Set2')
    # nn.NN(rpg_res2, rpg_test2, set2y_train, set2y_test, timestr, part='4', ds='Set2_RPG')
    # nn.NN(rpg_res2, rpg_test2, set2y_train, set2y_test, timestr, part='4', ds='Set2_RPG', final=True,hidden_layer_sizes=(50,100), alpha=0.1)
    #
    # # rps_res2, rps_test2
    # rps_res2, rps_test2 = fs.srp(set2X_train, set2y_train, set2X_test, set2y_test, timestr, n=6, part='3', ds='Set2')
    # nn.NN(rps_res2, rps_test2, set2y_train, set2y_test, timestr, part='4', ds='Set2_RPS')
    # nn.NN(rps_res2, rps_test2, set2y_train, set2y_test, timestr, part='4', ds='Set2_RPS', final=True,hidden_layer_sizes=(50,100,50), alpha=0.1)

        # Sean Crutchlow Today at 3:59 PM
        # For Part 4, I'm a bit confused. Since we're doing dimensionality reduction, do we apply the same DR algorithm
        # to both our training & test sets, or apply it once and split the data sets from there? Either way, we would still
        # be peeking into the test set & labels unfortunately.
        #
        # From earlier in the chat, I think I've been reading people are splitting post DR. Does this seem to be the consensus?
        #
        # I split pre dr and use model that is trained using train data to transform test data
        #
        # If you do what Elbek suggests correctly, then you will not be peeking into test data
        #
        # Ok. Thank you for letting me know. I was originally using the approach you mentioned transforming the test_X,
        # however, I'm dealing with some other issue were all my scores are the same across dimensionality.

    # sys.stdout.close()
    # sys.stdout = stdoutOrigin
    # stdoutOrigin = sys.stdout
    # sys.stdout = open("Part5"+timestr+".txt", "w")
    #
    # plt.close('all')

########################################################################################################################
    # 5.	Apply the clustering algorithms to the same dataset to which you just applied the dimensionality reduction
    # algorithms (you've probably already done this), treating the clusters as if they were new features. In other words,
    # treat the clustering algorithms as if they were dimensionality reduction algorithms. Again, rerun your neural network
    # learner on the newly projected data.

    # # RUN NN LEARNER ON OUTPUT FROM (1) ADDING THE FOUND CLUSTERS AS NEW COLUMNS
    # # •	Kmeans
    # # print("Set1")
    # # km_res1, km_test1 = km.km(set1X_train, set1y_train, set1X_test, set1y_test, timestr, k = 10, part = "5", ds = 'Set1')
    # # #print(np.shape(km_new_X1train),np.shape(set1y_train),np.shape(km_new_X1test),np.shape(set1y_test))
    # #
    # # enc = OneHotEncoder(handle_unknown='ignore')
    # # enc.fit(km_res1.reshape(-1, 1))
    # # onehot_km_res1 = enc.transform(km_res1.reshape(-1, 1)).toarray()
    # #
    # # enc.fit(km_test1.reshape(-1, 1))
    # # onehot_km_test1 = enc.transform(km_test1.reshape(-1, 1)).toarray()
    # #
    # # nn.NN(onehot_km_res1, onehot_km_test1, set1y_train, set1y_test, timestr, part='5', ds='Set1_KMoh')
    # # nn.NN(onehot_km_res1, onehot_km_test1, set1y_train, set1y_test, timestr, part='5', ds='Set1_KMoh', final=True,hidden_layer_sizes=(50,), alpha=0.1)
    # #
    # # km_new_X1train = (np.append(set1X_train, np.array(onehot_km_res1), axis=1))
    # # km_new_X1test = (np.append(set1X_test, np.array(onehot_km_test1), axis=1))
    # # nn.NN(km_new_X1train, km_new_X1test, set1y_train, set1y_test, timestr, part='5', ds='Set1_KM')
    # # nn.NN(km_new_X1train, km_new_X1test, set1y_train, set1y_test, timestr, part='5', ds='Set1_KM', final=True,hidden_layer_sizes=(50,100,50), alpha=0.5)
    #
    #
    # print("Set2")
    #
    #
    # km_res2, km_test2 = km.km(set2X_train, set2y_train, set2X_test, set2y_test, timestr, k = 2, part = "5", ds = 'Set2')
    # #print(np.shape(km_new_X2train),np.shape(set2y_train),np.shape(km_new_X2test),np.shape(set2y_test))
    #
    # enc = OneHotEncoder(handle_unknown='ignore')
    # enc.fit(km_res2.reshape(-1, 1))
    # onehot_km_res2 = enc.transform(km_res2.reshape(-1, 1)).toarray()
    #
    # enc.fit(km_test2.reshape(-1, 1))
    # onehot_km_test2 = enc.transform(km_test2.reshape(-1, 1)).toarray()
    #
    # nn.NN(onehot_km_res2, onehot_km_test2, set2y_train, set2y_test, timestr, part='5', ds='Set2_KMoh')
    # nn.NN(onehot_km_res2, onehot_km_test2, set2y_train, set2y_test, timestr, part='5', ds='Set2_KMoh', final=True,hidden_layer_sizes=(50,), alpha=0.1)
    #
    # km_new_X2train = (np.append(set2X_train, np.array(onehot_km_res2), axis=1))
    # km_new_X2test = (np.append(set2X_test, np.array(onehot_km_test2), axis=1))
    # nn.NN(km_new_X2train, km_new_X2test, set2y_train, set2y_test, timestr, part='5', ds='Set2_KM')
    # nn.NN(km_new_X2train, km_new_X2test, set2y_train, set2y_test, timestr, part='5', ds='Set2_KM', final=True,hidden_layer_sizes=(50,50,50), alpha=0.08)
    #
    #
    # # # •	Expectation Maximization
    # # em_res1, em_test1 = em.em(set1X_train, set1y_train, set1X_test, set1y_test, timestr, part = "1", ds = 'Set1')
    # # print("Set1")
    # # em_res1, em_test1 = em.em(set1X_train, set1y_train, set1X_test, set1y_test, timestr, k=2, part = "5", ds = 'Set1')
    # #
    # # enc.fit(em_res1.reshape(-1, 1))
    # # onehot_em_res1 = enc.transform(em_res1.reshape(-1, 1)).toarray()
    # #
    # # enc.fit(em_test1.reshape(-1, 1))
    # # onehot_em_test1 = enc.transform(em_test1.reshape(-1, 1)).toarray()
    # #
    # # nn.NN(onehot_em_res1, onehot_em_test1, set1y_train, set1y_test, timestr, part='5', ds='Set1_EMoh')
    # # nn.NN(onehot_em_res1, onehot_em_test1, set1y_train, set1y_test, timestr, part='5', ds='Set1_EMoh', final=True,hidden_layer_sizes=(50,), alpha=0.1)
    # #
    # # em_new_X1train = (np.append(set1X_train, np.array(onehot_em_res1), axis=1))
    # # em_new_X1test = (np.append(set1X_test, np.array(onehot_em_test1), axis=1))
    # # nn.NN(em_new_X1train, em_new_X1test, set1y_train, set1y_test, timestr, part='5', ds='Set1_EM')
    # # nn.NN(em_new_X1train, em_new_X1test, set1y_train, set1y_test, timestr, part='5', ds='Set1_EM', final=True,hidden_layer_sizes=(100,), alpha=0.5)
    #
    # print("Set2")
    # em_res2, em_test2 = em.em(set2X_train, set2y_train, set2X_test, set2y_test, timestr, k=4, part="5", ds='Set2')
    #
    # enc.fit(em_res2.reshape(-1, 1))
    # onehot_em_res2 = enc.transform(em_res2.reshape(-1, 1)).toarray()
    #
    # enc.fit(em_test2.reshape(-1, 1))
    # onehot_em_test2 = enc.transform(em_test2.reshape(-1, 1)).toarray()
    #
    # nn.NN(onehot_em_res2, onehot_em_test2, set2y_train, set2y_test, timestr, part='5', ds='Set2_EMoh')
    # nn.NN(onehot_em_res2, onehot_em_test2, set2y_train, set2y_test, timestr, part='5', ds='Set2_EMoh', final=True,hidden_layer_sizes=(50,), alpha=0.1)
    #
    # em_new_X2train = (np.append(set2X_train, np.array(onehot_em_res2), axis=1))
    # em_new_X2test = (np.append(set2X_test, np.array(onehot_em_test2), axis=1))
    # nn.NN(em_new_X2train, em_new_X2test, set2y_train, set2y_test, timestr, part='5', ds='Set2_EM')
    # nn.NN(em_new_X2train, em_new_X2test, set2y_train, set2y_test, timestr, part='5', ds='Set2_EM', final=True,hidden_layer_sizes=(50,), alpha=0.4)

    sys.stdout.close()
    sys.stdout = stdoutOrigin

    # Ashwin Ramanathan Yesterday at 10:55 AM
    # Besides plotting inertia/IC to do elbow method, and generating silhouette plots, are there any other plots to
    # create for the first 4 experiments?
    # 7 replies
    #
    # Brian Schendt  1 day ago
    # plotwise that sounds fine, but you'd probably want additional information, maybe in a table
    #
    # Jeremy  1 day ago
    # I was thinking it might be good to do some plotting in order to be able to analyze the clustering and DR results.
    # Though I suppose this could be done through tables and data rather than plotting
    #
    # Ashwin Ramanathan  1 day ago
    # Can you elaborate on what information we might want to convey in a table? I was thinking of using homogeneity
    # score but that also can be plotted vs the number of clusters
    #
    # Jeremy  24 hours ago
    # you shouldnt be plotting homogeneity score against the number of clusters. Homogeneity score requires the use of
    # the class labels so you should only be looking at it after you've chosen k
    #
    #
    # Brian Schendt  24 hours ago
    # Anything you think can answer the following questions:

    # Why did you get the clusters you did? Do they make "sense"?
    # If you used data that already had labels (for example data from a classification problem from assignment #1)
    # did the clusters line up with the labels?
    # Do they otherwise line up naturally? Why or why not?
    # Compare and contrast the different algorithms.
    # What sort of changes might you make to each of those algorithms to improve performance?
    # How much performance was due to the problems you chose?
    # Be creative and think of as many questions you can, and as many answers as you can. Take care to justify your
    # analysis with data explicitly.
    # Because the plots you mentioned, while necessarily, only answer how you choose k