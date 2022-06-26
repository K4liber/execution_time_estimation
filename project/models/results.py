import argparse
import logging
import os
import sys
from typing import Tuple

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


def get_avg_error_for_app(
        fraction_range: int,
        application_name: str,
        model_name: str,
        reduced: bool = False
) -> Tuple[float, float]:
    """

    :param fraction_range:
    :param application_name:
    :param model_name:
    :param reduced:
    :return: [full_data_error, mean_error_for_all_fractions]
    """

    app_id = app_name_to_id[application_name]
    model_details = get_model_details_for_algorithm(_application_name, _model_name, reduced=reduced)
    avg_errors_per_app = []

    for index, fraction in enumerate([round(1.0 - x / 10, 1) for x in range(fraction_range)]):
        model_file_name = get_model_file_name(model_details, application_name, fraction)
        model_file_path = os.path.join(os.path.join(ROOT_DIR, 'models', model_name), model_file_name)

        if not os.path.isfile(model_file_path):
            raise ValueError('Models not trained!')

        model = joblib.load(model_file_path)
        results_test_filepath = os.path.join(ROOT_DIR, '..', 'execution_results/results_test.csv')
        results_train_filepath = os.path.join(ROOT_DIR, '..', 'execution_results/results_train.csv')
        x_test, y_test = get_x_y(results_test_filepath, app_id, reduced=reduced)
        x_train, _ = get_x_y(results_train_filepath, app_id, reduced=reduced)

        if model_details.scale:
            init_scale_from_train_set(model_details, app_id)
        else:
            dismiss_scale()

        x_test_scaled = transform_x(x_test)
        y_predicted = inverse_transform_y(model.predict(x_test_scaled))
        y_list = list(y_test[DataFrameColumns.EXECUTION_TIME])
        _, errors_rel = get_errors(y_list, y_predicted)
        error_rel = sum(errors_rel) / len(errors_rel)
        avg_errors_per_app.append(error_rel)

        if index == 0:
            full_data_error = error_rel

    return full_data_error, sum(avg_errors_per_app) / len(avg_errors_per_app)


if __name__ == '__main__':
    _args = parser.parse_args()
    logger.info(_args)

    for _model_name in {'knn', 'pol', 'svr'}:
        logger.info(f'##### Model = {_model_name.upper()} #####')
        avg_errors_per_algorithm = []
        avg_errors_per_algorithm_reference = []
        full_data_avg_errors_per_algorithm = []
        full_data_avg_errors_per_algorithm_reference = []

        for _application_name in app_name_to_id.keys():
            full_data_error_per_app, avg_error_per_app \
                = get_avg_error_for_app(_args.frac, _application_name, _model_name)
            full_data_error_per_app_reduced, avg_error_per_app_reduced = \
                get_avg_error_for_app(_args.frac, _application_name, _model_name, reduced=True)
            logger.info(f'{_application_name.rjust(20)} errors = {round(avg_error_per_app, 1)} (original), '
                        f'{round(avg_error_per_app_reduced, 1)} (reduced)')
            avg_errors_per_algorithm.append(avg_error_per_app)
            avg_errors_per_algorithm_reference.append(avg_error_per_app_reduced)
            full_data_avg_errors_per_algorithm.append(full_data_error_per_app)
            full_data_avg_errors_per_algorithm_reference.append(full_data_error_per_app_reduced)

        logger.info(f'{"average".rjust(22)} errors = '
                    f'{round(sum(avg_errors_per_algorithm)/len(avg_errors_per_algorithm), 1)} (original), '
                    f'{round(sum(avg_errors_per_algorithm_reference)/len(avg_errors_per_algorithm_reference), 1)} '
                    f'(reduced)')
        logger.info(f'{"full training data".rjust(22)} errors = '
                    f'{round(sum(full_data_avg_errors_per_algorithm) / len(full_data_avg_errors_per_algorithm), 1)} '
                    f'(original), '
                    f'{round(sum(full_data_avg_errors_per_algorithm_reference) / len(full_data_avg_errors_per_algorithm_reference), 1)} '
                    f'(reduced)')
