import argparse
import os
from os.path import join
from os import getenv
import sys

import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import numpy as np

sys.path.append('.')

from project.models.knn.algorithm import AlgKNN
from project.models.svr.algorithm import AlgSVR
from project.models.xgb.algorithm import AlgXGB
from project.utils.app_ids import app_name_to_id
from project.utils.logger import logger
from project.definitions import ROOT_DIR
from project.models.data import (
    get_data_frame,
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

    for fraction in [round(1.0 - x/10, 1) for x in range(args.frac)]:
        df, df_err = get_data_frame(results_filepath, app_id, fraction)

        if df_err is not None:
            raise ValueError(f'data frame load err: {str(df_err)}')

        if args.reduced:
            x = df[DataFrameColumns.CPUS, DataFrameColumns.OVERALL_SIZE]
        else:
            x = df.loc[:, df.columns != DataFrameColumns.EXECUTION_TIME]

        y = df.loc[:, df.columns == DataFrameColumns.EXECUTION_TIME]

        if args.scale:
            init_scale(x, y)

        x_scaled = transform_x(x)
        y_scaled = transform_y(y)
        # ML start
        if args.alg == 'knn':
            algorithm = AlgKNN.get()
            param_grid = AlgKNN.get_params_grid()
        elif args.alg == 'svr':
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

        model.fit(x_scaled, np.ravel(y_scaled))
        y_predicted_scaled = model.predict(x_scaled)
        # ML end
        y_predicted = inverse_transform_y(y_predicted_scaled)
        y_list = list(y[DataFrameColumns.EXECUTION_TIME])
        errors_rel = []
        errors = []

        for index, y_pred in enumerate(y_predicted):
            y_pred = y_pred if y_pred > 0 else min(y_list)
            y_origin = y_list[index]
            error = abs(y_pred - y_origin)
            errors.append(error)
            error_rel = error * 100.0 / y_origin
            errors_rel.append(error_rel)

            if getenv("DEBUG") == "true":
                logger.info('pred: %s' % y_pred)
                logger.info('origin: %s' % y_origin)
                logger.info('error [s] = %s' % error)
                logger.info('error relative [percentage] = %s' % error_rel)

        logger.info('############### SUMMARY ##################')
        logger.info('model best params:')
        logger.info(model.best_params_)
        logger.info('training set length: %s' % len(y_list))
        logger.info('avg time [s] = %s' % str(sum(y_list) / len(y_list)))
        logger.info('avg error [s] = %s' % str(sum(errors) / len(errors)))
        logger.info('avg error relative [percentage] = %s' % str(sum(errors_rel) / len(errors_rel)))
        model_name = f'{args.app_name}_{fraction}'
        model_name = model_name + '_' + ('1' if args.scale else '0')
        model_name = model_name + '_' + ('1' if args.reduced else '0')
        model_path = os.path.join(ROOT_DIR, 'models', args.alg, model_name)

        with open(f'{model_path}.pkl', "w+b") as model_file:
            joblib.dump(model.best_estimator_, model_file, compress=1)
