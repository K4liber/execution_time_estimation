import math
import random
from os.path import join
import sys

sys.path.append('.')

import matplotlib.pyplot as plt
import matplotlib
from sklearn.neighbors import KNeighborsRegressor
import numpy as np

from project.definitions import ROOT_DIR


def main():
    figure_filepath = join(ROOT_DIR, 'tests/figures', 'knn.png')
    data_length = 200
    random_max = 0.2
    x = [4*math.pi*i/float(data_length) for i in range(data_length)]
    y = [math.sin(i) for i in x]
    # Plot signal
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(5.5, 4.5)
    plt.plot(x, y, label='signal = sin(explanatory variable)')
    x_data_point = x[:int(len(x)/2)]
    y_data_points = [i + random.uniform(-random_max, random_max) for i in y[:int(len(y)/2)]]
    model = KNeighborsRegressor(n_neighbors=3)
    x_train = np.array(x_data_point).reshape((-1, 1))
    model.fit(x_train, y_data_points)
    y_predict = model.predict(np.array(x).reshape((-1, 1)))
    plt.scatter(x_data_point, y_data_points, label='training data points (signal + noise)', c='y')
    plt.scatter(x, y_predict, label='predictions', c='g')
    plt.legend(loc='upper right')
    plt.xlabel('explanatory variable')
    plt.ylabel('response variable')
    plt.savefig(figure_filepath, bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    print('knn test')
    main()
