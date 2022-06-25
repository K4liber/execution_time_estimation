import argparse
import copy
import multiprocessing
import os
import sys
import time
from os.path import join
from typing import List, Dict, Type, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV

sys.path.append('.')

from project.models.interface import Algorithm
from project.models.common import get_model_details_for_algorithm, get_model_file_name, get_errors
from project.models.scale import init_scale, transform_y, inverse_transform_y, transform_x
from project.models.pol.algorithm import AlgPolynomialRegression
from project.models.knn.algorithm import AlgKNN
from project.models.svr.algorithm import AlgSVR
from project.models.xgb.algorithm import AlgXGB
from project.utils.app_ids import app_name_to_id
from project.utils.logger import logger
from project.definitions import ROOT_DIR
from project.models.data import (
    DataFrameColumns, get_x_y,
)

parser = argparse.ArgumentParser(description='Model training and validation.')
parser.add_argument('--app_name', required=True, type=str, help='app name')
parser.add_argument('--alg', required=True, type=str, help='algorithm')
parser.add_argument('--frac', required=False, default=1, type=int, help='number of fractions')


_name_to_algorithm: Dict[str, Type[Algorithm]] = {
    'knn': AlgKNN,
    'svr': AlgSVR,
    'xgb': AlgXGB,
    'pol': AlgPolynomialRegression,
}


def grid_search(
        algorithm: str,
        param_grid: dict
):
    return GridSearchCV(algorithm, param_grid=param_grid)


def run_grid_search(
        fraction: float,
        algorithm_name: str,
        application_name: str,
        x_scaled: pd.DataFrame,
        y_scaled: pd.DataFrame,
        y: List[float]
) -> Tuple[float, float, float]:
    """

    :param fraction:
    :param algorithm_name:
    :param application_name:
    :param x_scaled:
    :param y_scaled:
    :param y:
    :return: (fraction, training time [s], evaluation time [s])
    """
    model_details = get_model_details_for_algorithm(application_name, algorithm_name)

    if algorithm_name not in _name_to_algorithm:
        raise ValueError(f'"{algorithm_name}" algorithm not implemented')

    algorithm = _name_to_algorithm[algorithm_name].get()
    param_grid = _name_to_algorithm[algorithm_name].get_params_grid()
    model = grid_search(
        algorithm=algorithm,
        param_grid=param_grid,
    )
    x_train = x_scaled[:int(len(x_scaled) * fraction)]
    y_train = y_scaled[:int(len(y_scaled) * fraction)]
    training_time_start = time.time()
    model.fit(x_train, np.ravel(y_train))
    training_time = time.time() - training_time_start
    evaluation_time_start = time.time()
    y_predicted_scaled = model.predict(x_scaled)
    y_predicted = inverse_transform_y(y_predicted_scaled)
    evaluation_time = time.time() - evaluation_time_start
    errors, errors_rel = get_errors(y, y_predicted)
    logger.info('############### SUMMARY ##################')
    logger.info(f'algorithm: {algorithm_name}, app: {application_name}. fraction: {fraction}')
    logger.info('model best params:')
    logger.info(model.best_params_)
    logger.info('training set length: %s' % len(y_train))
    logger.info('avg time [s] = %s' % str(sum(y) / len(y)))
    logger.info('avg error [s] = %s' % str(sum(errors) / len(errors)))
    logger.info('avg error relative [percentage] = %s' % str(sum(errors_rel) / len(errors_rel)))
    model_name = get_model_file_name(model_details, application_name, fraction)
    model_path = os.path.join(ROOT_DIR, 'models', algorithm_name, model_name)

    with open(f'{model_path}', "w+b") as model_file:
        joblib.dump(model.best_estimator_, model_file, compress=1)

    return fraction, round(training_time, 5), round(evaluation_time, 5)


def run_grid_search_all_fractions(
        application_name: str,
        algorithm_name: str,
        frac: int
):
    model_details = get_model_details_for_algorithm(application_name, algorithm_name)
    app_id = app_name_to_id.get(application_name, None)

    if app_id is None:
        raise ValueError(
            f'missing app "{application_name}" from app map={app_name_to_id}'
        )

    results_train_filepath = join(ROOT_DIR, '..', 'execution_results/results_train.csv')
    x_train, y_train = get_x_y(results_train_filepath, app_id, model_details.reduced)

    if model_details.scale:
        init_scale(x_train, y_train)

    x_scaled = transform_x(x_train)
    y_scaled = transform_y(y_train)
    fractions = [round(1.0 - x / 10, 1) for x in range(frac)]
    grid_search_args = [
        (fraction, algorithm_name, application_name, copy.deepcopy(x_scaled), copy.deepcopy(y_scaled),
         copy.deepcopy(list(y_scaled[DataFrameColumns.EXECUTION_TIME]))) for fraction in fractions
    ]
    processes = min(len(fractions), 5)

    with multiprocessing.Pool(processes=processes) as pool:
        times_list = pool.starmap(run_grid_search, grid_search_args)
        times_filename = 'times' + ('_reduced' if model_details.reduced else '') + '.csv'
        times_filepath = join(ROOT_DIR, '..', 'execution_results', times_filename)

        with open(times_filepath, 'a') as times_file:
            for fraction, training_time, evaluation_time in times_list:
                values = [algorithm_name, application_name, fraction, training_time, evaluation_time]
                times_file.write(','.join([str(value) for value in values]) + '\n')


if __name__ == "__main__":
    args = parser.parse_args()
    logger.info(args)
    run_grid_search_all_fractions(args.app_name, args.alg, args.frac)
