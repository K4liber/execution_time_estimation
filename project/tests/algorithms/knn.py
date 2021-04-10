import math
import random

import matplotlib.pyplot as plt
import matplotlib
from sklearn.neighbors import KNeighborsRegressor
import numpy as np


def main():
    data_length = 200
    random_max = 0.2
    x = [8*math.pi*i/float(data_length) for i in range(data_length)]
    y = [math.sin(i) for i in x]
    x_data_point = x[:int(data_length/2)]
    y_data_points = [i + random.uniform(-random_max, random_max) for i in y[:int(data_length/2)]]
    model = KNeighborsRegressor(n_neighbors=3)
    x_train = np.array(x_data_point).reshape((-1, 1))
    model.fit(x_train, y_data_points)
    y_predict = model.predict(x_train)
    x_out = x[int(data_length / 2):]
    y_predict_out = model.predict(np.array(x_out).reshape((-1, 1)))
    # Plot results
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(10.5, 3.5)
    plt.plot(x, y, label='signal')
    plt.scatter(x_data_point, y_data_points, label='training data points', c='y')
    plt.scatter(x_data_point, y_predict, label='predictions in training range', c='g')
    plt.scatter(x_out, y_predict_out, label='predictions out of training range', c='r')
    plt.legend(loc='upper right')
    plt.title('KNN regression with out of training range predictions')
    plt.show()


if __name__ == '__main__':
    print('knn test')
    main()
