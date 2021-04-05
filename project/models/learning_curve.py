import argparse
import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np

sys.path.append('.')

from project.models.details import get_model_details, ModelDetails
from project.models.data import get_data_frame, DataFrameColumns
from project.models.scale import (
    init_scale,
    transform_x,
    transform_y,
    inverse_transform_y,
)
from project.utils.app_ids import app_name_to_id
from project.definitions import ROOT_DIR
from project.utils.logger import logger

parser = argparse.ArgumentParser(description='draw a learning curve')
parser.add_argument('--app_name', required=True, type=str, help='app name')
parser.add_argument('--alg', required=True, type=str, help='algorithm')
parser.add_argument('--scale', action=argparse.BooleanOptionalAction, default=False,
                    help='scale the data before learning')
parser.add_argument('--reduced', action=argparse.BooleanOptionalAction, default=False,
                    help='use only "CPUs" and "OVERALL_SIZE" features')


if __name__ == '__main__':
    args = parser.parse_args()
    logger.info(args)
    models_dir = os.path.join(ROOT_DIR, 'models', args.alg)

    if not os.path.isdir(models_dir):
        raise ValueError(f'"{models_dir}" is not a directory')

    app_id = app_name_to_id.get(args.app_name, None)

    if app_id is None:
        raise ValueError(f'missing app "{args.app_name}" from app map={str(app_name_to_id)}')

    results_filepath = os.path.join(ROOT_DIR, '..', 'execution_results/results.csv')
    df, df_err = get_data_frame(results_filepath, app_id)

    if df_err is not None:
        raise ValueError(f'data frame load err: {str(df_err)}')

    model_scheme = ModelDetails(args.app_name, 1.0, args.scale, args.reduced)

    if args.reduced:
        x = df[DataFrameColumns.CPUS, DataFrameColumns.OVERALL_SIZE]
    else:
        x = df.loc[:, df.columns != DataFrameColumns.EXECUTION_TIME]

    y = df.loc[:, df.columns == DataFrameColumns.EXECUTION_TIME]

    if args.scale:
        init_scale(x, y)

    x_scaled = transform_x(x)
    y_scaled = transform_y(y)
    frac = []
    model_error = []

    for filename in os.listdir(models_dir):
        if filename.startswith(args.app_name):
            model_details, err = get_model_details(filename)

            if err is not None:
                continue

            if model_scheme.the_same_run(model_details):
                logger.info(f'validating model "{filename}"')
                model = joblib.load(os.path.join(models_dir, filename))
                y_predicted_scaled = model.predict(x_scaled)
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

                    if os.getenv("DEBUG") == "true":
                        logger.info('pred: %s' % y_pred)
                        logger.info('origin: %s' % y_origin)
                        logger.info('error [s] = %s' % error)
                        logger.info('error relative [percentage] = %s' % error_rel)

                logger.info('############### SUMMARY ##################')
                logger.info(f'training data fraction: {model_details.frac}')
                logger.info('validation set length: %s' % len(y_list))
                logger.info('avg time [s] = %s' % str(sum(y_list) / len(y_list)))
                logger.info('avg error [s] = %s' % str(sum(errors) / len(errors)))
                error_rel = sum(errors_rel) / len(errors_rel)
                logger.info('avg error relative [percentage] = %s' % str(error_rel))
                frac.append(model_details.frac)
                model_error.append(error_rel)

    frac = np.array(frac)
    model_error = np.array(model_error)
    index_sorted = np.argsort(frac)
    x_plot_sorted = frac[index_sorted]
    y_plot_sorted = model_error[index_sorted]
    plt.plot(x_plot_sorted, y_plot_sorted)
    plt.show()
