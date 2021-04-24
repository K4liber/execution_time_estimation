import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np

sys.path.append('.')

from project.models.details import get_model_details, ModelDetails, get_model_name
from project.models.data import get_data_frame, DataFrameColumns, Const
from project.models.scale import (
    init_scale,
    transform_x,
    transform_y,
    inverse_transform_y, dismiss_scale,
)
from project.utils.app_ids import app_name_to_id, AppID
from project.definitions import ROOT_DIR
from project.utils.logger import logger


if __name__ == '__main__':
    knn_models_dir = os.path.join(ROOT_DIR, 'models', 'knn')

    if not os.path.isdir(knn_models_dir):
        raise ValueError(f'"{knn_models_dir}" is not a directory')

    svr_models_dir = os.path.join(ROOT_DIR, 'models', 'svr')

    if not os.path.isdir(svr_models_dir):
        raise ValueError(f'"{svr_models_dir}" is not a directory')

    results_filepath = os.path.join(ROOT_DIR, '..', 'execution_results/results_test.csv')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True)
    # fig.suptitle('Comparing the learning curves per module')
    app_id_to_axes = {
        AppID.XGBoostGridSearch: ax1,
        AppID.ImagesMerger: ax2,
        AppID.VideoSplitter: ax3,
        AppID.FaceRecogniser: ax4
    }

    for app_name, app_id in app_name_to_id.items():
        knn_model_scheme = ModelDetails(app_name, 1.0, False, False)
        knn_data_points = []
        knn_model_error = []
        df, df_err = get_data_frame(results_filepath, app_id)

        if df_err is not None:
            raise ValueError(f'data frame load err: {str(df_err)}')

        for filename in os.listdir(knn_models_dir):
            if filename.startswith(app_name):
                knn_model_details, err = get_model_details(filename)

                if err is not None:
                    continue

                if knn_model_scheme.the_same_run(knn_model_details):
                    logger.info(f'validating model "{filename}"')
                    errors_rel = []
                    errors = []
                    model = joblib.load(os.path.join(knn_models_dir, filename))

                    if knn_model_details.reduced:
                        x = df[DataFrameColumns.CPUS, DataFrameColumns.OVERALL_SIZE]
                    else:
                        x = df.loc[:, df.columns != DataFrameColumns.EXECUTION_TIME]

                    y = df.loc[:, df.columns == DataFrameColumns.EXECUTION_TIME]

                    if knn_model_details.scale:
                        init_scale(x, y)
                    else:
                        dismiss_scale()

                    x_scaled = transform_x(x)
                    y_scaled = transform_y(y)
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
                    logger.info(f'training data fraction: {knn_model_details.frac}')
                    logger.info('validation set length: %s' % len(y_list))
                    logger.info('avg error [s] = %s' % str(sum(errors) / len(errors)))
                    error_rel = sum(errors_rel) / len(errors_rel)
                    logger.info('avg error relative [percentage] = %s' % str(error_rel))
                    knn_data_points.append(knn_model_details.frac * Const.TRAINING_SAMPLES)
                    knn_model_error.append(error_rel)

        knn_data_points = np.array(knn_data_points)
        knn_model_error = np.array(knn_model_error)
        knn_index_sorted = np.argsort(knn_data_points)
        knn_x_plot_sorted = knn_data_points[knn_index_sorted]
        knn_y_plot_sorted = knn_model_error[knn_index_sorted]
        app_id_to_axes[app_id].plot(knn_x_plot_sorted, knn_y_plot_sorted, label=f'knn')
        app_id_to_axes[app_id].set_title(app_name)

        svr_model_scheme = ModelDetails(app_name, 1.0, True, False)
        svr_data_points = []
        svr_model_error = []

        for filename in os.listdir(svr_models_dir):
            if filename.startswith(app_name):
                svr_model_details, err = get_model_details(filename)

                if err is not None:
                    logger.warning(str(err))
                    continue

                if svr_model_scheme.the_same_run(svr_model_details):
                    logger.info(f'validating model "{filename}"')
                    errors_rel = []
                    errors = []
                    model = joblib.load(os.path.join(svr_models_dir, filename))
                    df, df_err = get_data_frame(results_filepath, app_id)

                    if df_err is not None:
                        raise ValueError(f'data frame load err: {str(df_err)}')

                    if svr_model_details.reduced:
                        x = df[DataFrameColumns.CPUS, DataFrameColumns.OVERALL_SIZE]
                    else:
                        x = df.loc[:, df.columns != DataFrameColumns.EXECUTION_TIME]

                    y = df.loc[:, df.columns == DataFrameColumns.EXECUTION_TIME]

                    if svr_model_details.scale:
                        results_train_filepath = os.path.join(ROOT_DIR, '..', 'execution_results/results_train.csv')
                        df_train, df_err = get_data_frame(results_train_filepath, app_id)

                        if df_err is not None:
                            raise ValueError(f'data frame load err: {str(df_err)}')

                        init_scale(
                            df_train.loc[:, df_train.columns != DataFrameColumns.EXECUTION_TIME],
                            df_train.loc[:, df_train.columns == DataFrameColumns.EXECUTION_TIME]
                        )
                    else:
                        dismiss_scale()

                    x_scaled = transform_x(x)
                    y_scaled = transform_y(y)
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
                    logger.info(f'training data fraction: {svr_model_details.frac}')
                    logger.info('validation set length: %s' % len(y_list))
                    logger.info('avg error [s] = %s' % str(sum(errors) / len(errors)))
                    error_rel = sum(errors_rel) / len(errors_rel)
                    logger.info('avg error relative [percentage] = %s' % str(error_rel))
                    svr_data_points.append(svr_model_details.frac * Const.TRAINING_SAMPLES)
                    svr_model_error.append(error_rel)

        svr_data_points = np.array(svr_data_points)
        svr_model_error = np.array(svr_model_error)
        svr_index_sorted = np.argsort(svr_data_points)
        svr_x_plot_sorted = svr_data_points[svr_index_sorted]
        svr_y_plot_sorted = svr_model_error[svr_index_sorted]
        app_id_to_axes[app_id].plot(svr_x_plot_sorted, svr_y_plot_sorted, label=f'svr')

    for ax in fig.get_axes():
        ax.label_outer()
        ax.set_ylim(0, 200)
        ax.legend()

    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel('training samples quantity')
    plt.ylabel('regression relative error [%]')
    plt.tight_layout()
    fig_path = os.path.join(ROOT_DIR, 'models', 'figures', 'learning_curve_multi' + '.png')
    plt.savefig(fig_path, bbox_inches='tight', pad_inches=0)
