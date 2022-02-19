import argparse
import logging
import os
import sys

import joblib

sys.path.append('.')

from project.definitions import ROOT_DIR
from project.models.common import get_model_details_for_algorithm, get_model_file_name, init_scale_from_train_set, \
    get_errors
from project.models.data import get_x_y, DataFrameColumns
from project.models.scale import transform_x, inverse_transform_y, dismiss_scale
from project.utils.app_ids import app_name_to_id

logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser(description='Model training and validation.')
parser.add_argument('--frac', required=False, default=10, type=int, help='number of fractions')


if __name__ == '__main__':
    args = parser.parse_args()
    logger.info(args)

    for model_name in {'knn', 'pol', 'svr'}:
        logger.info(f'##### Model = {model_name.upper()} #####')
        avg_errors_per_algorithm = []

        for application_name, app_id in app_name_to_id.items():
            model_details = get_model_details_for_algorithm(application_name, model_name)
            avg_errors_per_app = []

            for fraction in [round(1.0 - x / 10, 1) for x in range(args.frac)]:
                model_file_name = get_model_file_name(model_details, application_name, fraction)
                model_file_path = os.path.join(os.path.join(ROOT_DIR, 'models', model_name), model_file_name)

                if not os.path.isfile(model_file_path):
                    raise ValueError('Models not trained!')

                model = joblib.load(model_file_path)
                results_test_filepath = os.path.join(ROOT_DIR, '..', 'execution_results/results_test.csv')
                results_train_filepath = os.path.join(ROOT_DIR, '..', 'execution_results/results_train.csv')
                x_test, y_test = get_x_y(results_test_filepath, app_id)
                x_train, _ = get_x_y(results_train_filepath, app_id)

                if model_details.scale:
                    init_scale_from_train_set(model_details, app_id)
                else:
                    dismiss_scale()

                x_test_scaled = transform_x(x_test)
                x_train_scaled = transform_x(x_train)
                y_predicted = inverse_transform_y(model.predict(x_test_scaled))
                y_list = list(y_test[DataFrameColumns.EXECUTION_TIME])
                _, errors_rel = get_errors(y_list, y_predicted)
                avg_errors_per_app.append(sum(errors_rel)/len(errors_rel))

            avg_error_per_app = sum(avg_errors_per_app)/len(avg_errors_per_app)
            logger.info(f'"{application_name}" errors = {round(avg_error_per_app, 1)}')
            avg_errors_per_algorithm.append(avg_error_per_app)

        logger.info(f'{model_name} average error = '
                    f'{round(sum(avg_errors_per_algorithm)/len(avg_errors_per_algorithm), 1)}')
