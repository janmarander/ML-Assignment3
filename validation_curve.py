# https://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html#sphx-glr-auto-examples-model-selection-plot-validation-curve-py

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import validation_curve
import time

def make_validation(X,y, data_set_name, model, model_name, param_name, param_range):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    #print(timestr)

    train_scores, test_scores = validation_curve(
        model, X, y, param_name=param_name, param_range=param_range,
        scoring="accuracy", cv = 10, n_jobs=1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title('%s Validation Curve for %s, %s' % (model_name, data_set_name, param_name))
    plt.xlabel(param_name)
    plt.ylabel("Accuracy")
    # plt.ylim(0.0, 1.1)
    lw = 2

    if model_name == "Neural Network" and param_name=='nn__hidden_layer_sizes':
        labels = []
        for tup in param_range:
            labels.append(str(tup))
        param_list_range = list(range(0,len(param_range)))
        plt.plot(param_list_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
        plt.fill_between(param_list_range, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.2,
                         color="darkorange", lw=lw)
        plt.plot(param_list_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
        plt.fill_between(param_list_range, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.2,
                         color="navy", lw=lw)
        plt.xticks(param_list_range,labels)
        #plt.xticks(ticks, labels)
    else:
        plt.plot(param_range, train_scores_mean, label="Training score",
                     color="darkorange", lw=lw)
        plt.fill_between(param_range, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.2,
                         color="darkorange", lw=lw)
        plt.plot(param_range, test_scores_mean, label="Cross-validation score",
                     color="navy", lw=lw)
        plt.fill_between(param_range, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.2,
                         color="navy", lw=lw)
    plt.legend(loc="best")
    plt.savefig('%s Validation Curve %s'% (data_set_name,model_name) +timestr+'.png' )
    plt.show()