import math
import random
from os.path import join
import sys

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

sys.path.append('.')

from project.models.grid_search import grid_search
from project.models.svr.algorithm import AlgSVR
from project.definitions import ROOT_DIR


def main():
    random.seed(1)
    figure_filepath = join(ROOT_DIR, 'tests/figures', 'svr_out_of_range.png')
    data_length = 100
    random_max = 0.05
    train_frac = 3/4
    x = [30*i/float(data_length) for i in range(data_length)]
    y = [0.1*math.exp(0.1*i) for i in x]
    # Plot signal
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(4.5, 3.5)
    plt.plot(x, y, label='signal = ~exp(x)', color='black')
    x_data_point = x[:int(len(x)*train_frac)]
    y_data_points = [i + random.uniform(-random_max, random_max) for i in y[:int(len(y)*train_frac)]]
    algorithm = AlgSVR.get()
    param_grid = AlgSVR.get_params_grid()
    model = grid_search(
        algorithm=algorithm,
        param_grid=param_grid,
    )
    x_train = np.array(x_data_point).reshape((-1, 1))
    model.fit(x_train, y_data_points)
    y_predict = model.predict(np.array(x).reshape((-1, 1)))
    plt.scatter(x_data_point, y_data_points, alpha=0.8, label='training points (signal + noise)', c='g')
    plt.scatter(x, y_predict, alpha=0.5, label='svr predictions', c='y')
    plt.ylim([-0.1, 2.5])
    plt.xlim([0, 32])
    plt.legend(loc='upper left')
    plt.xlabel('x')
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
