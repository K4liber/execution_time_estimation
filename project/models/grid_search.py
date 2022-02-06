import argparse
import copy
import multiprocessing
import os
from os.path import join
from os import getenv
import sys
from typing import List

import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV
import numpy as np

sys.path.append('.')

from project.models.common import get_model_details_for_algorithm
from project.models.scale import init_scale, transform_y, inverse_transform_y, transform_x
from project.models.pol.algorithm import AlgPolynomialRegression
from project.models.knn.algorithm import AlgKNN, KNNParam
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


def grid_search(algorithm, param_grid):
    return GridSearchCV(algorithm, param_grid=param_grid)


def run_grid_search(
        fraction: float,
        algorithm_name: str,
        application_name: str,
        x_scaled: pd.DataFrame,
        y_scaled: pd.DataFrame,
        y: List[float]
):
    model_details = get_model_details_for_algorithm(application_name, algorithm_name)

    if algorithm_name == 'knn':
        algorithm = AlgKNN.get()
        param_grid = AlgKNN.get_params_grid({KNNParam.MAX_N: int(len(x_scaled) * fraction * 0.8) - 1})
    elif algorithm_name == 'svr':
        algorithm = AlgSVR.get()
        param_grid = AlgSVR.get_params_grid()
    elif algorithm_name == 'xgb':
        algorithm = AlgXGB.get()
        param_grid = AlgXGB.get_params_grid()
    elif algorithm_name == 'pol':
        algorithm = AlgPolynomialRegression.get()
        param_grid = AlgPolynomialRegression.get_params_grid()
    else:
        raise ValueError(f'"{algorithm_name}" algorithm not implemented')

    model = grid_search(
        algorithm=algorithm,
        param_grid=param_grid,
    )
    x_train = x_scaled[:int(len(x_scaled) * fraction)]
    y_train = y_scaled[:int(len(y_scaled) * fraction)]
    model.fit(x_train, np.ravel(y_train))
    y_predicted_scaled = model.predict(x_scaled)
    # ML end
    y_predicted = inverse_transform_y(y_predicted_scaled)
    errors_rel = []
    errors = []

    for index, y_pred in enumerate(y_predicted):
        y_pred = y_pred if y_pred > 0 else min(y)
        y_origin = y[index]
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
    logger.info(f'algorithm: {algorithm_name}, app: {application_name}. fraction: {fraction}')
    logger.info('model best params:')
    logger.info(model.best_params_)
    logger.info('training set length: %s' % len(y_train))
    logger.info('avg time [s] = %s' % str(sum(y) / len(y)))
    logger.info('avg error [s] = %s' % str(sum(errors) / len(errors)))
    logger.info('avg error relative [percentage] = %s' % str(sum(errors_rel) / len(errors_rel)))
    model_name = f'{application_name}_{fraction}'
    model_name = model_name + '_' + ('1' if model_details.scale else '0')
    model_name = model_name + '_' + ('1' if model_details.reduced else '0')
    model_path = os.path.join(ROOT_DIR, 'models', algorithm_name, model_name)

    with open(f'{model_path}.pkl', "w+b") as model_file:
        joblib.dump(model.best_estimator_, model_file, compress=1)


def run_grid_search_all_fractions(application_name: str, algorithm_name: str, frac: int):
    model_details = get_model_details_for_algorithm(application_name, algorithm_name)
    app_id = app_name_to_id.get(application_name, None)

    if app_id is None:
        raise ValueError(
            f'missing app "{application_name}" from app map={app_name_to_id}'
        )

    results_filepath = join(ROOT_DIR, '..', 'execution_results/results_train.csv')
    df, df_err = get_data_frame(results_filepath, app_id)

    if df_err is not None:
        raise ValueError(f'data frame load err: {df_err}')

    if model_details.reduced:
        x = df[DataFrameColumns.CPUS, DataFrameColumns.OVERALL_SIZE]
    else:
        x = df.loc[:, df.columns != DataFrameColumns.EXECUTION_TIME]

    y = df.loc[:, df.columns == DataFrameColumns.EXECUTION_TIME]

    if model_details.scale:
        init_scale(x, y)

    x_scaled = transform_x(x)
    y_scaled = transform_y(y)
    grid_search_args = [
        (fraction, algorithm_name, application_name, copy.deepcopy(x_scaled), copy.deepcopy(y_scaled),
         copy.deepcopy(list(y[DataFrameColumns.EXECUTION_TIME])))
        for fraction in [round(1.0 - x / 10, 1) for x in range(frac)]
    ]

    with multiprocessing.Pool(processes=5) as pool:
        pool.starmap(run_grid_search, grid_search_args)


if __name__ == "__main__":
    args = parser.parse_args()
    logger.info(args)
    run_grid_search_all_fractions(args.app_name, args.alg, args.frac)
