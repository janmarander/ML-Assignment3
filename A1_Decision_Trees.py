
# Decision Trees. For the decision tree, you should implement or steal a decision tree algorithm (and by
# "implement or steal" I mean "steal"). Be sure to use some form of pruning. You are not required to use
# information gain (for example, there is something called the GINI index that is sometimes used) to split
# attributes, but you should describe whatever it is that you do use.
# https://napsterinblue.github.io/notes/machine_learning/datasets/make_classification/
# https://learning.oreilly.com/library/view/machine-learning-with/9781789343700/97578e59-8914-4dc9-8ee9-9e6307cafe84.xhtml
# https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html

import sklearn as sk
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import plotly
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from make_plots import plot_learning_curve
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


def DT(set1X_train, set1X_test, set1y_train, set1y_test,timestr,part='3', ds='Set1'):
    print("DT")
    #timestr = time.strftime("%Y%m%d-%H%M%S")
    print(timestr)

    # Setting up the scaling pipeline
    pipeline_order = [('scaler', StandardScaler()), ('dt', DecisionTreeClassifier(criterion = 'gini', random_state = 50))]
    DTpipe = Pipeline(pipeline_order)
    # Fitting the classfier to the scaled dataset
    dt_classifier_scaled1 = DTpipe.fit(set1X_train, set1y_train)

    # Extracting the score
    print(dt_classifier_scaled1.score(set1X_train, set1y_train))
    # Testing accuracy on the test data
    dt_classifier_scaled1.score(set1X_test, set1y_test)
    print(dt_classifier_scaled1.score(set1X_test, set1y_test))

    # Creating a grid of different hyperparameters
    grid_params = {
        'dt__max_depth': [3, 4, 5, 6, 8, 12, 14, 16, 18, 20],
        'dt__ccp_alpha': [.00005, .0002, .0003, .0004, .0005, .001]
        # 'dt__min_samples_leaf': [0.0001,0.001, 0.005,0.02,0.04, 0.06, 0.08, .1, .2, .5]
    }

    # Building a 10 fold Cross Validated GridSearchCV object
    grid_object = GridSearchCV(estimator=dt_classifier_scaled1, param_grid=grid_params, scoring='accuracy', cv=8,
                               n_jobs=-1)

    # Fitting the grid to the training data
    grid_object.fit(set1X_train, set1y_train)

    # Extracting the best parameters
    print(grid_object.best_params_)
    rf_best1 = grid_object.best_estimator_

    print(rf_best1)

    print("Final DT scores on train/test data " + "Part " + str(part) + ", " + ds + ":")
    start = time.time()
    rf_best1.fit(set1X_train, set1y_train)
    end = time.time()
    print('Train time: ', end - start)
    print('Train score: ', rf_best1.score(set1X_train, set1y_train))
    start = time.time()
    print('Test score: ', rf_best1.score(set1X_test, set1y_test))
    end = time.time()
    print('Test time: ', end - start)

    # title = "Decision Trees"
    # plt = plot_learning_curve(rf_best1, title, set1X_train, set1y_train, axes=None, ylim=None, cv=None, n_jobs=-1,
    #                           train_sizes=np.linspace(.1, 1.0, 5))
    #
    # # fig, axes = plt.subplots(3, 2, figsize=(10, 15))
    # plt.savefig('Data1 DT LC'+timestr+'.png')
    # plt.show()
    #
    # # Building a 10 fold Cross Validated GridSearchCV object
    # grid_object = GridSearchCV(estimator=dt_classifier_scaled2, param_grid=grid_params, scoring='accuracy', cv=8,
    #                            n_jobs=-1)
    #
    # # Fitting the grid to the training data
    # grid_object.fit(set2X_train, set2y_train)
    #
    # # Extracting the best parameters
    # print(grid_object.best_params_)
    # rf_best2 = grid_object.best_estimator_
    #
    # title = "Decision Trees"
    # plt = plot_learning_curve(rf_best2, title, set2X_train, set2y_train, axes=None, ylim=None, cv=None, n_jobs=-1,
    #                           train_sizes=np.linspace(.1, 1.0, 5))
    #
    # # fig, axes = plt.subplots(3, 2, figsize=(10, 15))
    # plt.savefig('Data2 DT LC'+timestr+'.png')
    # plt.show()
    #
    # data1 = {'X_train': set1X_train, 'X_test': set1X_test, 'y_train': set1y_train, 'y_test': set1y_test}
    # data2 = {'X_train': set2X_train, 'X_test': set2X_test, 'y_train': set2y_train, 'y_test': set2y_test}
    #
    # # GENERATE MODEL COMPLEXITY CURVES!!!!
    #
    # # TUNED PARAMETERS:
    # min_samples_leaf = 0.001
    # ccp_alpha = (1e-1) * 10 ** -0.5
    # max_depth = 14
    #
    # # NEED TWO COMPLEXITY CURVES OF HYPERPARAMETERS:
    # # 1. Depth (max_depth)
    # # 2. ccp_alpha?
    # max_depths = [3, 4, 5, 6, 8, 12, 14, 16, 18, 20]
    # vc.make_validation(set1X_train, set1y_train, 'Set1', rf_best1, "Decision Tree", 'dt__max_depth', max_depths)
    #
    # vc.make_validation(set2X_train, set2y_train, 'Set2', rf_best2, "Decision Tree", 'dt__max_depth', max_depths)
    #
    # pipeline_order = [('scaler', StandardScaler()),
    #                   ('dt', DecisionTreeClassifier(max_depth=30, criterion='gini', random_state=50))]
    # DTpipe_param1 = Pipeline(pipeline_order)
    #
    # pipeline_order = [('scaler', StandardScaler()),
    #                   ('dt', DecisionTreeClassifier(max_depth=30, criterion='gini', random_state=50))]
    # DTpipe_param2 = Pipeline(pipeline_order)
    #
    # ccp_alphas = [.0001, .0002, .0004, .0006, .0008, .001, .0015]
    # vc.make_validation(set1X_train, set1y_train, 'Set1', rf_best1, "Decision Tree", 'dt__ccp_alpha',
    #                    ccp_alphas)
    #
    # vc.make_validation(set2X_train, set2y_train, 'Set2', rf_best2, "Decision Tree", 'dt__ccp_alpha',
    #                    ccp_alphas)
    #
    #
    # # configurations = [
    # #     # {'estimator': DecisionTreeClassifier,
    # #     #  'tuned_params': {'criterion': 'gini', 'random_state': 50, 'min_samples_leaf': min_samples_leaf}, #, 'ccp_alpha': ccp_alpha},
    # #     #  'changing_param': 'ccp_alpha',
    # #     #  'changing_param_values': [0, .00005, .0002, .0003, .0004, .025],
    # #     #  'complexity_label': 'ccp_alpha',
    # #     #  'complexity_computer': lambda x: x.ccp_alpha,
    # #     #  'prediction_performance_computer': accuracy_score,
    # #     #  'prediction_performance_label': 'accuracy',
    # #     #  'postfit_hook': lambda x: x,
    # #     #  'data': data1,
    # #     #  'n_samples': 30},
    # #     {'estimator': DecisionTreeClassifier,
    # #      'tuned_params': {'criterion': 'gini', 'random_state': 50,'ccp_alpha': ccp_alpha},  # , 'ccp_alpha': ccp_alpha},
    # #      'changing_param': 'min_samples_leaf',
    # #      'changing_param_values': [0.001, 0.005,0.02,0.04, 0.06, 0.08, .1, .2, .4, .5],
    # #      'complexity_label': 'min_samples_leaf',
    # #      'complexity_computer': lambda x: x.min_samples_leaf,
    # #      'prediction_performance_computer': accuracy_score,
    # #      'prediction_performance_label': 'accuracy',
    # #      'postfit_hook': lambda x: x,
    # #      'data': data1,
    # #      'n_samples': 30},
    # #     {'estimator': DecisionTreeClassifier,
    # #      'tuned_params': {'criterion': 'gini', 'random_state': 50, 'min_samples_leaf': min_samples_leaf},# , 'ccp_alpha': ccp_alpha},
    # #      'changing_param': 'max_depth',
    # #      'changing_param_values': [1,2,3,4,5,6,8,12,14,16],
    # #      'complexity_label': 'max_depth',
    # #      'complexity_computer': lambda x: x.max_depth,
    # #      'prediction_performance_computer': accuracy_score,
    # #      'prediction_performance_label': 'accuracy',
    # #      'postfit_hook': lambda x: x,
    # #      'data': data1,
    # #      'n_samples': 30},
    # # ]
    # #
    # # mc.plotMC(configurations)
    #
    # #best params after MC tuning:
    # newrf_best1 = Pipeline(steps=[('scaler', StandardScaler()),
    #                 ('dt',
    #                  DecisionTreeClassifier(max_depth=8,
    #                                         ccp_alpha=0.001,
    #                                         random_state=50))])
    #
    # title = "Decision Trees data1"
    # plt = plot_learning_curve(newrf_best1, title, set1X_train, set1y_train, axes=None, ylim=None, cv=None, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5))
    # plt.savefig('Data1 Decision Trees Learning Curve' + timestr + '.png')
    # plt.show()
    #
    # print("Final scores on train/test data Set1:")
    # start = time.time()
    # newrf_best1.fit(set1X_train, set1y_train)
    # end = time.time()
    # print('Train time: ', end - start)
    # print('Train score: ', newrf_best1.score(set1X_train, set1y_train))
    # start = time.time()
    # print('Test score: ', newrf_best1.score(set1X_test, set1y_test))
    # end = time.time()
    # print('Test time: ', end - start)
    #
    # newrf_best2 = Pipeline(steps=[('scaler', StandardScaler()),
    #                               ('dt',
    #                  DecisionTreeClassifier(max_depth=16,
    #                                         ccp_alpha=0.0004,
    #                                         random_state=50))])
    # title = "Decision Trees data2"
    # plt = plot_learning_curve(newrf_best2, title, set2X_train, set2y_train, axes=None, ylim=None, cv=None, n_jobs=-1,
    #                           train_sizes=np.linspace(.1, 1.0, 5))
    # plt.savefig('Data2 Decision Trees Learning Curve' + timestr + '.png')
    # plt.show()
    #
    # print("Final scores on train/test data Set2:")
    # start = time.time()
    # newrf_best2.fit(set2X_train, set2y_train)
    # end = time.time()
    # print('Train time: ', end - start)
    # print('Train score: ', newrf_best2.score(set2X_train, set2y_train))
    # start = time.time()
    # print('Test score: ', newrf_best2.score(set2X_test, set2y_test))
    # end = time.time()
    # print('Test time: ', end - start)
    #
