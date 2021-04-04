# Neural Networks. For the neural network you should implement or steal your favorite kind of network and
# training algorithm. You may use networks of nodes with as many layers as you like and any activation function
# you see fit.

#This has probably been mentioned but is the intent for part 5 with NN to use only the cluster labels as input features
# to the NN or use the cluster labels in addition to the existing features?
# John Miller  3 hours ago
# In the OH threads they have said either is fine

import sklearn as sk
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import plotly
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from make_plots import plot_learning_curve
from make_plots import plot_loss_curve
import model_complexity as mc
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.svm import NuSVR
from sklearn.metrics import hamming_loss
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import validation_curve as vc
import json
import time
import timeit


def NN(set1X_train, set1X_test, set1y_train, set1y_test, timestr, part='3', ds='Set1', final = False, hidden_layer_sizes = [], alpha = 0.1):
    print("ANN")
    # timestr = time.strftime("%Y%m%d-%H%M%S")
    # print(timestr)

    #array.reshape(-1, 1)

    if(final):
        # best params after MC tuning:
        newrf_best1 = Pipeline(steps=[('scaler', StandardScaler()),
                                      ('nn',
                                       MLPClassifier(max_iter=1000,
                                                     learning_rate='constant',
                                                     solver='sgd',
                                                     activation='tanh',
                                                     hidden_layer_sizes=hidden_layer_sizes,
                                                     alpha=alpha
                                                     ))])
        title = "Part " + str(part) + ", " + ds + ' Neural Network Learning Curve'
        plt = plot_learning_curve(newrf_best1, title, set1X_train, set1y_train, axes=None, ylim=None, cv=None,
                                  n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5))
        plt.savefig("Part " + str(part) + ", " + ds + ' Neural Network Learning Curve' + timestr + '.png')
        plt.show()

        print("Final scores on train/test data " + "Part " + str(part) + ", " + ds + ":")
        start = time.time()
        newrf_best1.fit(set1X_train, set1y_train)
        end = time.time()
        print('Train time: ', end - start)
        print('Train score: ', newrf_best1.score(set1X_train, set1y_train))
        start = time.time()
        print('Test score: ', newrf_best1.score(set1X_test, set1y_test))
        end = time.time()
        print('Test time: ', end - start)
    else:

        # Setting up the scaling pipeline
        pipeline_order = [('scaler', StandardScaler()), ('nn', MLPClassifier(max_iter=1000,activation = 'tanh', learning_rate='constant'))]
        NNpipe = Pipeline(pipeline_order)
        # Fitting the classfier to the scaled dataset
        nn_classifier_scaled1 = NNpipe.fit(set1X_train, set1y_train)

        # Extracting the score
        print(nn_classifier_scaled1.score(set1X_train, set1y_train))
        # Testing accuracy on the test data
        nn_classifier_scaled1.score(set1X_test, set1y_test)
        print(nn_classifier_scaled1.score(set1X_test, set1y_test))


        # Creating a grid of different hyperparameters

        grid_params = {
            'nn__hidden_layer_sizes': [(50,50,50), (50,100,50), (50,100), (100,)],
            # 'nn__activation': ['tanh', 'relu'],
            # 'nn__solver': ['sgd', 'adam'],
            'nn__alpha': [0.001, 0.05, 0.1, .5, .8] #[0.0001, 0.05],
            #'nn__learning_rate': ['constant','adaptive'],
        }

        # Building a 10 fold Cross Validated GridSearchCV object
        grid_object = GridSearchCV(estimator=nn_classifier_scaled1, param_grid=grid_params, scoring='accuracy', cv=8,
                                   n_jobs=-1)

        # Fitting the grid to the training data
        grid_object.fit(set1X_train, set1y_train)

        # Extracting the best parameters
        print(grid_object.best_params_)
        rf_best1 = grid_object.best_estimator_

        print(rf_best1)

        title = ds + " Neural Network"
        plt = plot_learning_curve(rf_best1, title, set1X_train, set1y_train, axes=None, ylim=None, cv=None, n_jobs=None,
                                  train_sizes=np.linspace(.1, 1.0, 5))

        #fig, axes = plt.subplots(3, 2, figsize=(10, 15))
        plt.savefig("Part " + str(part)+ ", " + ds+ ' Neural Network LC'+timestr+'.png')
        plt.show()

        plt = plot_loss_curve(rf_best1, title, set1X_train, set1y_train, axes=None, ylim=None, cv=None, n_jobs=None,
                                  train_sizes=np.linspace(.1, 1.0, 5))
        plt.savefig("Part " + str(part)+ ", " + ds+ ' Neural Network LossCurve'+timestr+'.png')
        plt.show()

       # GENERATE MODEL COMPLEXITY CURVES!!!!

        # TUNED PARAMETERS:


        grid_params = {
            'nn__hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
            'nn__activation': ['tanh', 'relu'],
            'nn__solver': ['sgd', 'adam'],
            'nn__alpha': [0.0001, 0.05],
            'nn__learning_rate': ['constant','adaptive'],
        }
        # NEED TWO COMPLEXITY CURVES OF HYPERPARAMETERS:
        # 1. Hidden Layers, Width, Depth????
        # 2. ???? pick something else

        nn__hidden_layer_sizes= [(20,20,20), (50,50,50), (50,100,50), (50,100), (100,), (50,), (10,)]
        #nn__hidden_layer_sizes = [(100, 500, 100), (50, 100, 50), (50, 50, 50), (20, 20, 20), (100, 500), (50, 100),  (500,), (100,), (50,), (20,)]
        vc.make_validation(set1X_train, set1y_train, ds, rf_best1, "Neural Network", 'nn__hidden_layer_sizes', nn__hidden_layer_sizes)

        pipeline_order = [('scaler', StandardScaler()),
                          ('nn', MLPClassifier(max_iter=500))]
        NNpipe_param1 = Pipeline(pipeline_order)

        pipeline_order = [('scaler', StandardScaler()),
                          ('nn', MLPClassifier(max_iter=500))]
        NNpipe_param2 = Pipeline(pipeline_order)

        nn__alphas =  [0.01, 0.05, 0.08, 0.1, .15, .3, .45, .5, .8]
        vc.make_validation(set1X_train, set1y_train, ds, rf_best1, "Neural Network", 'nn__alpha',
                           nn__alphas)

        #best params after MC tuning:
        # newrf_best1 = Pipeline(steps=[('scaler', StandardScaler()),
        #                 ('nn',
        #                  MLPClassifier(max_iter=500,
        #                                learning_rate = 'constant',
        #                                solver = 'sgd',
        #                                activation = 'tanh',
        #                                hidden_layer_sizes=(50,100,50),
        #                                alpha=0.1))])
        # title = "Neural Network " + ds
        # plt = plot_learning_curve(newrf_best1, title, set1X_train, set1y_train, axes=None, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5))
        # plt.savefig("Part " + str(part)+ ", " + ds + ' Neural Network Learning Curve' + timestr + '.png')
        # plt.show()
        #
        #
        # print("Final scores on train/test data " + "Part " + str(part)+ ", " + ds+ ":")
        # start = time.time()
        # newrf_best1.fit(set1X_train, set1y_train)
        # end = time.time()
        # print('Train time: ', end - start)
        # print('Train score: ', newrf_best1.score(set1X_train, set1y_train))
        # start = time.time()
        # print('Test score: ', newrf_best1.score(set1X_test, set1y_test))
        # end = time.time()
        # print('Test time: ', end - start)

