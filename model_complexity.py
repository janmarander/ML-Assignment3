# https://scikit-learn.org/stable/auto_examples/applications/plot_model_complexity_influence.html

# Authors: Eustache Diemert <eustache@diemert.fr>
#          Maria Telenczuk <https://github.com/maikia>
#          Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: BSD 3 clause

import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.svm import NuSVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import hamming_loss


# # Initialize random generator
# np.random.seed(0)
#
#         """Generate regression/classification data."""
#         if case == 'regression':
#             X, y = datasets.load_diabetes(return_X_y=True)
#     elif case == 'classification':
#         X, y = datasets.fetch_20newsgroups_vectorized(subset='all',
#                                                       return_X_y=True)
#     X, y = shuffle(X, y)
#     offset = int(X.shape[0] * 0.8)
#     X_train, y_train = X[:offset], y[:offset]
#     X_test, y_test = X[offset:], y[offset:]
#
#     data = {'X_train': X_train, 'X_test': X_test, 'y_train': y_train,
#             'y_test': y_test}
#     return data
#
#
# regression_data = generate_data('regression')
# classification_data = generate_data('classification')

def benchmark_influence(conf):
    """
    Benchmark influence of `changing_param` on both MSE and latency.
    """
    prediction_times = []
    prediction_powers = []
    complexities = []
    for param_value in conf['changing_param_values']:
        conf['tuned_params'][conf['changing_param']] = param_value
        estimator = conf['estimator'](**conf['tuned_params'])

        print("Benchmarking %s" % estimator)
        estimator.fit(conf['data']['X_train'], conf['data']['y_train'])
        print(estimator.get_depth())
        conf['postfit_hook'](estimator)
        complexity = conf['complexity_computer'](estimator)
        complexities.append(complexity)
        start_time = time.time()
        for _ in range(conf['n_samples']):
            y_pred = estimator.predict(conf['data']['X_test'])
        elapsed_time = (time.time() - start_time) / float(conf['n_samples'])
        prediction_times.append(elapsed_time)
        pred_score = conf['prediction_performance_computer'](conf['data']['y_test'], y_pred)
        prediction_powers.append(pred_score)
        print("Complexity: %d | %s: %.4f | Pred. Time: %fs\n" % (
            complexity, conf['prediction_performance_label'], pred_score,
            elapsed_time))
    return prediction_powers, prediction_times, complexities

def _count_nonzero_coefficients(estimator):
    a = estimator.coef_.toarray()
    return np.count_nonzero(a)


def plot_influence(conf, mse_values, prediction_times, complexities):
    """
    Plot influence of model complexity on both accuracy and latency.
    """

    fig = plt.figure()
    fig.subplots_adjust(right=0.75)

    # first axes (prediction error)
    ax1 = fig.add_subplot(111)
    line1 = ax1.plot(complexities, mse_values, c='tab:blue', ls='-')[0]
    ax1.set_xlabel('Model Complexity (%s)' % conf['complexity_label'])
    y1_label = conf['prediction_performance_label']
    ax1.set_ylabel(y1_label)

    ax1.spines['left'].set_color(line1.get_color())
    ax1.yaxis.label.set_color(line1.get_color())
    ax1.tick_params(axis='y', colors=line1.get_color())

    # second axes (latency)
    ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)
    line2 = ax2.plot(complexities, prediction_times, c='tab:orange', ls='-')[0]
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    y2_label = "Time (s)"
    ax2.set_ylabel(y2_label)
    ax1.spines['right'].set_color(line2.get_color())
    ax2.yaxis.label.set_color(line2.get_color())
    ax2.tick_params(axis='y', colors=line2.get_color())

    plt.legend((line1, line2), ("prediction accuracy", "latency"),
               loc='upper right')

    plt.title("Influence of varying '%s' on %s" % (conf['changing_param'],
                                                   conf['estimator'].__name__))


def plotMC(configurations):
    for conf in configurations:
        prediction_performances, prediction_times, complexities = benchmark_influence(conf)
        plot_influence(conf, prediction_performances, prediction_times, complexities)
    plt.show()


