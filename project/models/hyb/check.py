import argparse
import logging
import os

import joblib

from project.definitions import ROOT_DIR
from project.models.common import get_model_details_for_algorithm, get_model_file_name, init_scale_from_train_set, \
    get_errors
from project.models.data import get_x_y, DataFrameColumns
from project.models.scale import transform_x, inverse_transform_y
from project.utils.app_ids import app_name_to_id

logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser(description='Model training and validation.')
parser.add_argument('--app_name', required=True, type=str, help='app name')
parser.add_argument('--frac', required=False, default=10, type=int, help='number of fractions')

if __name__ == '__main__':
    args = parser.parse_args()
    logger.info(args)
    application_name = args.app_name
    app_id = app_name_to_id.get(application_name, None)

    if app_id is None:
        raise ValueError(
            f'missing app "{application_name}" from app map={app_name_to_id}'
        )

    knn_model_details = get_model_details_for_algorithm(application_name, 'knn')
    svr_model_details = get_model_details_for_algorithm(application_name, 'svr')

    for fraction in [round(1.0 - x / 10, 1) for x in range(args.frac)]:
        knn_model_file_name = get_model_file_name(knn_model_details, application_name, fraction)
        knn_model_file_path = os.path.join(os.path.join(ROOT_DIR, 'models', 'knn'), knn_model_file_name)
        svr_model_file_name = get_model_file_name(knn_model_details, application_name, fraction)
        svr_model_file_path = os.path.join(os.path.join(ROOT_DIR, 'models', 'svr'), svr_model_file_name)

        if not all(lambda file_path: os.path.isfile(file_path) for file_path in [
            svr_model_file_path, knn_model_file_path]):
            raise ValueError('Models not trained!')

        knn_model = joblib.load(knn_model_file_path)
        svr_model = joblib.load(svr_model_file_path)
        results_test_filepath = os.path.join(ROOT_DIR, '..', 'execution_results/results_test.csv')
        results_train_filepath = os.path.join(ROOT_DIR, '..', 'execution_results/results_train.csv')
        x_test, y_test = get_x_y(results_test_filepath, app_id)
        x_train, _ = get_x_y(results_train_filepath, app_id)

        if knn_model_details.scale:
            init_scale_from_train_set(knn_model_details, app_id)

        x_test_scaled = transform_x(x_test)
        x_train_scaled = transform_x(x_train)
        min_x_train = x_train_scaled.min(axis=0)
        max_x_train = x_train_scaled.max(axis=0)
        min_x_test = x_test_scaled.min(axis=0)
        max_x_test = x_test_scaled.max(axis=0)
        y_predicted_knn = inverse_transform_y(knn_model.predict(x_test_scaled))
        y_predicted_svr = inverse_transform_y(svr_model.predict(x_test_scaled))
        y_predicted_hyb = []

        for idx in range(len(y_test.index)):
            x_scaled = x_test_scaled[idx]
            is_in_range = all(min_x_train[col_index] <= x_scaled[col_index] <= max_x_train[col_index]
                              for col_index in range(len(x_scaled)))

            if not is_in_range:
                print('yolo')

            y_origin = y_test.iloc[idx]
            y_knn = y_predicted_knn[idx]
            y_svr = y_predicted_svr[idx]
            y_hybrid = y_knn if is_in_range else y_svr
            y_predicted_hyb.append(y_hybrid)

        y_list = list(y_test[DataFrameColumns.EXECUTION_TIME])
        knn_errors, knn_errors_rel = get_errors(y_list, y_predicted_knn)
        svr_errors, svr_errors_rel = get_errors(y_list, y_predicted_svr)
        hyb_errors, hyb_errors_rel = get_errors(y_list, y_predicted_hyb)
        logger.info(f'##### SUMMARY application = {application_name}, fraction = {fraction} #####')
        logger.info(f'KNN errors = {sum(knn_errors_rel)/len(knn_errors_rel)}')
        logger.info(f'SVR errors = {sum(svr_errors_rel) / len(svr_errors_rel)}')
        logger.info(f'Hybrid errors = {sum(hyb_errors_rel) / len(hyb_errors_rel)}')
