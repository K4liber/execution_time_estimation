import os
from typing import List, Tuple, Optional

import joblib
import numpy as np

from project.definitions import ROOT_DIR, is_reduced
from project.models.data import get_data_frame, DataFrameColumns, Const, get_x_y, REDUCED_FEATURES
from project.models.details import ModelDetails
from project.models.scale import init_scale, dismiss_scale, transform_x, inverse_transform_y
from project.utils.app_ids import app_name_to_id
from project.utils.logger import logger

_algorithm_to_color = {
    'svr': 'y',
    'knn': 'b',
    'pol': 'g'
}


def get_model_details_for_algorithm(application_name: str, algorithm: str, fraction: float = 1.0,
                                    reduced: Optional[bool] = None) -> ModelDetails:
    reduced = is_reduced() if reduced is None else reduced
    algorithm_to_model_details = {
        'svr': ModelDetails(application_name, fraction, True, reduced),
        'knn': ModelDetails(application_name, fraction, True, reduced),
        'pol': ModelDetails(application_name, fraction, False, reduced)
    }
    return algorithm_to_model_details[algorithm]


def get_color(algorithm: str) -> str:
    return _algorithm_to_color[algorithm]


def init_scale_from_train_set(model_details: ModelDetails, app_id: int):
    results_train_filepath = os.path.join(ROOT_DIR, '..', 'execution_results/results_train.csv')
    df_train, df_err = get_data_frame(results_train_filepath, app_id)

    if df_err is not None:
        raise ValueError(f'data frame load err: {df_err}')

    init_scale(
        df_train.loc[:, df_train.columns != DataFrameColumns.EXECUTION_TIME if not model_details.reduced
                        else REDUCED_FEATURES],
        df_train.loc[:, df_train.columns == DataFrameColumns.EXECUTION_TIME]
    )


def plot_app_learning_curve(application_name: str, algorithm_name: str, ax, results_filepath: Optional[str]):
    algorithm_dir = os.path.join(ROOT_DIR, 'models', algorithm_name)

    if not os.path.isdir(algorithm_dir):
        raise ValueError(f'"{algorithm_dir}" is not a directory')

    app_id = app_name_to_id.get(application_name, None)

    if app_id is None:
        raise ValueError(
            f'missing app "{application_name}" from app map={app_name_to_id}'
        )

    results_test_filepath = os.path.join(ROOT_DIR, '..', 'execution_results/results_test.csv')
    data_points = []
    model_error = []

    for fraction in [round(1.0 - x / 10, 1) for x in range(10)]:
        model_details = get_model_details_for_algorithm(application_name, algorithm_name, fraction)
        model_file_name = get_model_file_name(model_details, application_name, fraction)
        model_file_path = os.path.join(algorithm_dir, model_file_name)

        if not os.path.isfile(model_file_path):
            continue

        logger.info(f'validating model "{model_file_name}"')
        model = joblib.load(model_file_path)
        x_test, y_test = get_x_y(results_test_filepath, app_id, model_details.reduced)

        if model_details.scale:
            init_scale_from_train_set(model_details, app_id)
        else:
            dismiss_scale()

        x_scaled = transform_x(x_test)
        y_predicted_scaled = model.predict(x_scaled)
        y_predicted = inverse_transform_y(y_predicted_scaled)
        y_list = list(y_test[DataFrameColumns.EXECUTION_TIME])
        errors, errors_rel = get_errors(y_list, y_predicted)

        if results_filepath:
            with open(results_filepath, 'a') as results_file:
                results_file.write("\n".join([str(value) for value in errors_rel]))
                results_file.write("\n")

        logger.info('############### SUMMARY ##################')
        logger.info(model.get_params())
        logger.info(f'training data fraction: {fraction}')
        logger.info('validation set length: %s' % len(y_list))
        logger.info('avg error [s] = %s' % str(sum(errors) / len(errors)))
        error_rel = sum(errors_rel) / len(errors_rel)
        logger.info('avg error relative [percentage] = %s' % str(error_rel))
        data_points.append(fraction * Const.TRAINING_SAMPLES)
        model_error.append(error_rel)

    data_points = np.array(data_points)
    model_error = np.array(model_error)
    index_sorted = np.argsort(data_points)
    x_plot_sorted = data_points[index_sorted]
    y_plot_sorted = model_error[index_sorted]
    ax.plot(x_plot_sorted, y_plot_sorted, label=algorithm_name, color=get_color(algorithm_name))
    ax.set_title(application_name)


def get_model_file_name(model_details: ModelDetails, application_name: str, fraction: float) -> str:
    model_name = f'{application_name}_{fraction}'
    model_name = model_name + '_' + ('1' if model_details.scale else '0')
    return model_name + '_' + ('1' if model_details.reduced else '0') + '.pkl'


def get_errors(y_real: List[float], y_predicted: List[float]) -> Tuple[List[float], List[float]]:
    errors = []
    errors_rel = []
    avg_real = sum(y_real) / len(y_real)

    for index, y_pred in enumerate(y_predicted):
        y_pred = y_pred if y_pred > 0 else min(y_real)  # Execution time cannot be lower than zero
        y_origin = y_real[index]
        error = abs(y_pred - y_origin)
        errors.append(error)
        error_rel = error * 100.0 / abs(avg_real)
        errors_rel.append(error_rel)

        if os.getenv("DEBUG") == "true":
            logger.info('pred: %s' % y_pred)
            logger.info('origin: %s' % y_origin)
            logger.info('error [s] = %s' % error)
            logger.info('error relative [percentage] = %s' % error_rel)

    return errors, errors_rel


def get_name_suffix(reduced: Optional[bool] = None):
    _is_reduced = is_reduced() if reduced is None else reduced
    return '_reduced' if _is_reduced else ''
