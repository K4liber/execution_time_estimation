import argparse
import os
from os.path import join
from os import getenv
import sys

import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

sys.path.append('.')

from project.models.svr.algorithm import AlgSVR
from project.models.xgb import AlgXGB
from project.utils.app_ids import app_name_to_id
from project.utils.logger import logger
from project.definitions import ROOT_DIR
from project.models.data import (
    get_data_frame,
    get_training_test_split,
    DataFrameColumns,
)

scale_x = None
scale_y = None
parser = argparse.ArgumentParser(description='Model training and validation.')
parser.add_argument('--app_name', required=True, type=str, help='app name')
parser.add_argument('--alg', required=True, type=str, help='algorithm')
parser.add_argument('--frac', required=False, default=1, type=int, help='number of fractions')
parser.add_argument('--scale', action=argparse.BooleanOptionalAction, help='scale the data before learning')
parser.add_argument('--reduced', action=argparse.BooleanOptionalAction,
                    help='use only "CPUs" and "OVERALL_SIZE" features')


def init_scale(x: any, y: any):
    global scale_x
    global scale_y
    scale_x = StandardScaler().fit(x)
    scale_y = StandardScaler().fit(y)


def transform_x(x: any) -> any:
    if scale_x is not None:
        return scale_x.transform(x)
    else:
        return x


def transform_y(y: any) -> any:
    if scale_y is not None:
        return scale_y.transform(y)
    else:
        return y


def inverse_transform_y(y: any) -> any:
    if scale_y is not None:
        return scale_y.inverse_transform(y)
    else:
        return y


def grid_search(algorithm, param_grid):
    return GridSearchCV(algorithm, param_grid=param_grid)


if __name__ == "__main__":
    args = parser.parse_args()
    logger.info(args)
    app_id = app_name_to_id.get(args.app_name, None)

    if app_id is None:
        raise ValueError(f'missing app "{args.app_name}" from app map={str(app_name_to_id)}')

    results_filepath = join(ROOT_DIR, '..', 'execution_results/results.csv')

    for fraction in [1.0 - x/10 for x in range(args.frac)]:
        df, df_err = get_data_frame(results_filepath, app_id, fraction)

        if df_err is not None:
            raise ValueError(f'data frame load err: {str(df_err)}')

        columns = None

        if args.reduced:
            columns = [DataFrameColumns.CPUS, DataFrameColumns.OVERALL_SIZE]

        x, y, x_train, x_test, y_train, y_test = get_training_test_split(df, columns)

        if args.scale:
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
        # ML start
        if args.alg == 'svr':
            algorithm = AlgSVR.get()
            param_grid = AlgSVR.get_params_grid()
        elif args.alg == 'xgb':
            algorithm = AlgXGB.get()
            param_grid = AlgXGB.get_params_grid()
        else:
            raise ValueError(f'"{args.alg}" algorithm not implemented')

        model = grid_search(
            algorithm=algorithm,
            param_grid=param_grid,
        )

        model.fit(x_train_scaled, np.ravel(y_train_scaled))
        z_svr = model.predict(x_scaled)
        z_svr_test = model.predict(x_test_scaled)
        # ML end
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
                logger.info('pred: %s' % z_pred)
                logger.info('origin: %s' % z_origin)
                logger.info('error [s] = %s' % error)
                logger.info('error relative [percentage] = %s' % error_rel)

        logger.info('############### SUMMARY ##################')
        logger.info('model best params:')
        logger.info(model.best_params_)
        logger.info('training set length: %s' % len(y_train_list))
        logger.info('test set length: %s' % len(y_test_list))
        logger.info('avg time [s] = %s' % str(sum(y_test_list) / len(y_test_list)))
        logger.info('avg error [s] = %s' % str(sum(errors) / len(errors)))
        logger.info('avg error relative [percentage] = %s' % str(sum(errors_rel) / len(errors_rel)))
        model_name = f'{args.app_name}_{fraction}'
        model_name = model_name + '_' + ('1' if args.scale else '0')
        model_name = model_name + '_' + ('1' if args.reduced else '0')
        model_path = os.path.join(ROOT_DIR, 'models', args.alg, model_name)
        joblib.dump(model.best_estimator_, f'{model_path}.pkl', compress=1)

        if getenv("DEBUG") == "true":
            # Plot prediction surface
            z_svr_inverse = inverse_transform_y(z_svr)
            x_plot = x[DataFrameColumns.OVERALL_SIZE].to_numpy()
            y_plot = x[DataFrameColumns.CPUS].to_numpy()
            ax.plot_trisurf(x_plot, y_plot, z_svr_inverse, alpha=0.5)
            plt.margins()
            plt.gcf().autofmt_xdate()
            ax.legend()
            plt.show()
