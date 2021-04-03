import argparse
from os.path import join
from os import getenv
import sys

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
import numpy as np

sys.path.append('.')

from project.utils.app_ids import app_name_to_id
from project.utils.logger import logger
from project.definitions import ROOT_DIR
from project.models.data import (
    get_data_frame,
    get_training_test_split, DataFrameColumns, FEATURE_NAMES,
)

parser = argparse.ArgumentParser(description='Model training and tests.')
parser.add_argument('--app_name', required=True, type=str, help='app name')

USE_SCALE = True
scale_x = None
scale_y = None


def init_scale(x: any, y: any):
    global scale_x
    global scale_y
    scale_x = StandardScaler().fit(x)
    scale_y = StandardScaler().fit(y)


def transform_x(x: any) -> any:
    if USE_SCALE:
        return scale_x.transform(x)
    else:
        return x


def transform_y(y: any) -> any:
    if USE_SCALE:
        return scale_y.transform(y)
    else:
        return y


def inverse_transform_y(y: any) -> any:
    if USE_SCALE:
        return scale_y.inverse_transform(y)
    else:
        return y


if __name__ == "__main__":
    args = parser.parse_args()
    logger.info(args)
    app_id = app_name_to_id.get(args.app_name, None)

    if app_id is None:
        raise ValueError(f'missing app "{args.app_name}" from app map={str(app_name_to_id)}')

    results_filepath = join(ROOT_DIR, '..', 'execution_results/results.csv')
    df, df_err = get_data_frame(results_filepath, app_id)

    if df_err is not None:
        raise ValueError(f'data frame load err: {str(df_err)}')

    x, y, x_train, x_test, y_train, y_test = get_training_test_split(
        df, [DataFrameColumns.CPUS, DataFrameColumns.OVERALL_SIZE])

    if USE_SCALE:
        init_scale(x, y)

    x_test_scaled = transform_x(x_test)
    x_scaled = transform_x(x)
    x_train_scaled = transform_x(x_train)
    y_train_scaled = transform_y(y_train)
    scaled_x_train_df = pd.DataFrame(x_train_scaled, columns=x.columns)
    scaled_y_train_df = pd.DataFrame(y_train_scaled, columns=y.columns)
    x_plot_train = x_train[DataFrameColumns.OVERALL_SIZE]
    y_plot_train = x_train[DataFrameColumns.CPUS]
    z_plot_train = y_train[DataFrameColumns.EXECUTION_TIME]
    x_plot_test = x_test[DataFrameColumns.OVERALL_SIZE]
    y_plot_test = x_test[DataFrameColumns.CPUS]
    z_plot_test = y_test[DataFrameColumns.EXECUTION_TIME]
    # plot data points
    ax = plt.axes(projection='3d')
    ax.set_xlabel('total size [B]')
    ax.set_ylabel('cpus')
    ax.set_zlabel('time [s]')
    ax.scatter(x_plot_train, y_plot_train, z_plot_train, c='#2ca02c', alpha=1, label='training points')
    ax.scatter(x_plot_test, y_plot_test, z_plot_test, label='test points', c='#cc0000', alpha=1)
    # ML
    gamma_min = 0.001
    epsilon_min = 0.00001
    C_min = 10.0
    model = GridSearchCV(SVR(kernel='rbf'),
                         param_grid={
                             "gamma": [gamma_min * 2 ** x for x in range(11)],
                             "epsilon": [epsilon_min * 2 ** x for x in range(11)],
                             "C": [C_min * 2 ** x for x in range(15)],
                         })
    model.fit(x_train_scaled, np.ravel(y_train_scaled))
    logger.info('model best params:')
    logger.info(model.best_params_)
    z_svr = model.predict(x_scaled)
    z_svr_test = model.predict(x_test_scaled)
    z_svr_test_inverse = inverse_transform_y(z_svr_test)
    y_test_list = list(y_test[DataFrameColumns.EXECUTION_TIME])
    y_train_list = list(y_train[DataFrameColumns.EXECUTION_TIME])
    errors_rel = []
    errors = []

    for index, z_pred in enumerate(z_svr_test_inverse):
        z_pred = z_pred if z_pred > 0 else min(y_train_list)
        z_origin = y_test_list[index]
        error = abs(z_pred - z_origin)
        errors.append(error)
        error_rel = error * 100.0 / z_origin
        errors_rel.append(error_rel)

        if getenv("DEBUG") == "true":
            print('pred: %s' % z_pred)
            print('origin: %s' % z_origin)
            print('error [s] = %s' % error)
            print('error relative [percentage] = %s' % error_rel)

    print('############### SUMMARY ##################')
    print('test set length: %s' % len(y_test_list))
    print('avg time [s] = %s' % str(sum(y_test_list) / len(y_test_list)))
    print('avg error [s] = %s' % str(sum(errors) / len(errors)))
    print('avg error relative [percentage] = %s' % str(sum(errors_rel) / len(errors_rel)))
    # Plot prediction surface
    z_svr_inverse = inverse_transform_y(z_svr)
    x_plot = x[DataFrameColumns.OVERALL_SIZE].to_numpy()
    y_plot = x[DataFrameColumns.CPUS].to_numpy()
    ax.plot_trisurf(x_plot, y_plot, z_svr_inverse, alpha=0.5)
    # ML end
    plt.margins()
    plt.gcf().autofmt_xdate()
    ax.legend()
    plt.show()
    # Plot execution_time in the feature function
    # Plot 2d charts
    for feature_name in FEATURE_NAMES:
        plt.clf()
        x_plot = x[feature_name].to_numpy()
        index_sorted = np.argsort(x_plot)
        x_plot_sorted = x_plot[index_sorted]
        y_plot_sorted = z_svr_inverse[index_sorted]
        plt.scatter(x_plot_sorted, y_plot_sorted)
        plt.title(f'Execution time [s] in the {feature_name} function')
        plt.savefig(f'{feature_name}.jpg')
