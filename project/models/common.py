import os

import joblib
import numpy as np

from project.definitions import ROOT_DIR
from project.models.data import get_data_frame, DataFrameColumns, Const
from project.models.details import ModelDetails, get_model_details
from project.models.scale import init_scale, dismiss_scale, transform_x, inverse_transform_y
from project.utils.app_ids import app_name_to_id
from project.utils.logger import logger


_algorithm_to_color = {
    'svr': 'y',
    'knn': 'b',
    'pol': 'g'
}


def get_model_details_for_algorithm(application_name: str, algorithm: str) -> ModelDetails:
    algorithm_to_model_details = {
        'svr': ModelDetails(application_name, 1.0, True, False),
        'knn': ModelDetails(application_name, 1.0, True, False),
        'pol': ModelDetails(application_name, 1.0, False, False)
    }
    return algorithm_to_model_details[algorithm]


def get_color(algorithm: str) -> str:
    return _algorithm_to_color[algorithm]


def plot_app_learning_curve(application_name: str, algorithm_name: str, ax):
    algorithm_dir = os.path.join(ROOT_DIR, 'models', algorithm_name)

    if not os.path.isdir(algorithm_dir):
        raise ValueError(f'"{algorithm_dir}" is not a directory')

    app_id = app_name_to_id.get(application_name, None)

    if app_id is None:
        raise ValueError(
            f'missing app "{application_name}" from app map={app_name_to_id}'
        )

    results_filepath = os.path.join(ROOT_DIR, '..', 'execution_results/results_test.csv')
    model_scheme = get_model_details_for_algorithm(application_name, algorithm_name)
    data_points = []
    model_error = []

    for filename in sorted(os.listdir(algorithm_dir)):
        if filename.startswith(application_name):
            model_details, err = get_model_details(filename)

            if err is not None:
                logger.warning(str(err))
                continue

            if model_scheme.the_same_run(model_details):
                logger.info(f'validating model "{filename}"')
                errors_rel = []
                errors = []
                model = joblib.load(os.path.join(algorithm_dir, filename))
                df, df_err = get_data_frame(results_filepath, app_id)

                if df_err is not None:
                    raise ValueError(f'data frame load err: {df_err}')

                if model_details.reduced:
                    x = df[DataFrameColumns.CPUS, DataFrameColumns.OVERALL_SIZE]
                else:
                    x = df.loc[:, df.columns != DataFrameColumns.EXECUTION_TIME]

                y = df.loc[:, df.columns == DataFrameColumns.EXECUTION_TIME]

                if model_details.scale:
                    results_train_filepath = os.path.join(ROOT_DIR, '..', 'execution_results/results_train.csv')
                    df_train, df_err = get_data_frame(results_train_filepath, app_id)

                    if df_err is not None:
                        raise ValueError(f'data frame load err: {df_err}')

                    init_scale(
                        df_train.loc[:, df_train.columns != DataFrameColumns.EXECUTION_TIME],
                        df_train.loc[:, df_train.columns == DataFrameColumns.EXECUTION_TIME]
                    )
                else:
                    dismiss_scale()

                x_scaled = transform_x(x)
                y_predicted_scaled = model.predict(x_scaled)
                y_predicted = inverse_transform_y(y_predicted_scaled)
                y_list = list(y[DataFrameColumns.EXECUTION_TIME])

                for index, y_pred in enumerate(y_predicted):
                    y_pred = y_pred if y_pred > 0 else min(y_list)
                    y_origin = y_list[index]
                    error = abs(y_pred - y_origin)
                    errors.append(error)
                    error_rel = error * 100.0 / y_origin
                    errors_rel.append(error_rel)

                if os.getenv("DEBUG") == "true":
                    logger.info('pred: %s' % y_pred)
                    logger.info('origin: %s' % y_origin)
                    logger.info('error [s] = %s' % error)
                    logger.info('error relative [percentage] = %s' % error_rel)

                logger.info('############### SUMMARY ##################')
                logger.info(model.get_params())
                logger.info(f'training data fraction: {model_details.frac}')
                logger.info('validation set length: %s' % len(y_list))
                logger.info('avg error [s] = %s' % str(sum(errors) / len(errors)))
                error_rel = sum(errors_rel) / len(errors_rel)
                logger.info('avg error relative [percentage] = %s' % str(error_rel))
                data_points.append(model_details.frac * Const.TRAINING_SAMPLES)
                model_error.append(error_rel)

    data_points = np.array(data_points)
    model_error = np.array(model_error)
    index_sorted = np.argsort(data_points)
    x_plot_sorted = data_points[index_sorted]
    y_plot_sorted = model_error[index_sorted]
    ax.plot(x_plot_sorted, y_plot_sorted, label=algorithm_name, color=get_color(algorithm_name))
    ax.set_title(application_name)
