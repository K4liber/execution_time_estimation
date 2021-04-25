import math
import random
from os.path import join
import sys

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

sys.path.append('.')

from project.utils.logger import logger
from project.models.grid_search import grid_search
from project.models.svr.algorithm import AlgSVR
from project.definitions import ROOT_DIR


def main():
    random.seed(1)
    figure_filepath = join(ROOT_DIR, 'tests/figures', 'svr_gamma.png')
    data_length = 100
    random_max = 0.5
    x = [3*math.pi*i/float(data_length) for i in range(data_length)]
    y = [5 + 2*i + 6*random.uniform(-random_max, random_max) + 4*math.sin(1.5*i) for i in x]
    # Plot signal
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(4.5, 3.5)
    x_data_point = x
    y_data_points = [i + random.uniform(-random_max, random_max) for i in y]
    param_grid = AlgSVR.get_params_grid()
    param_grid['gamma'] = [0.1]
    param_grid['C'] = [64000.0]
    param_grid['epsilon'] = [1e-06]
    model = grid_search(
        algorithm=AlgSVR.get(),
        param_grid=param_grid,
    )
    param_grid_2 = AlgSVR.get_params_grid()
    param_grid_2['gamma'] = [10]
    param_grid_2['C'] = [64000.0]
    param_grid_2['epsilon'] = [1e-06]
    model_2 = grid_search(
        algorithm=AlgSVR.get(),
        param_grid=param_grid_2,
    )
    x_train = np.array(x_data_point).reshape((-1, 1))
    model.fit(x_train, y_data_points)
    logger.info(model.best_estimator_)
    model_2.fit(x_train, y_data_points)
    y_predict = model.predict(np.array(x).reshape((-1, 1)))
    y_predict_2 = model_2.predict(np.array(x).reshape((-1, 1)))
    plt.plot(x, y_predict, 'r-', alpha=1, label='gamma = 0.1')
    plt.plot(x, y_predict_2, 'b--', alpha=1, label='gamma = 10')
    plt.scatter(x_data_point, y_data_points, alpha=0.7, label='training points', c='g')
    plt.legend(loc='upper left')
    plt.xlabel('predictor variable')
    plt.ylabel('response variable')
    plt.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,
        left=False,  # ticks along the bottom edge are off
        right=False,
        labelbottom=False,
        labeltop=False,
        labelright=False,
        labelleft=False
    )
    plt.savefig(figure_filepath, bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    print('knn test')
    main()
