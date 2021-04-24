import argparse
import os
from os.path import join
from os import getenv
import sys

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import tkinter
import matplotlib
matplotlib.use('TkAgg')

sys.path.append('.')

from project.models.details import get_model_filepath, ModelDetails, get_model_name
from project.models.scale import init_scale, transform_x, transform_y, inverse_transform_y
from project.utils.app_ids import app_name_to_id
from project.utils.logger import logger
from project.definitions import ROOT_DIR
from project.models.data import (
    get_data_frame,
    get_training_test_split,
    DataFrameColumns,
)

parser = argparse.ArgumentParser(description='Model training and validation.')
parser.add_argument('--app_name', required=True, type=str, help='app name')
parser.add_argument('--alg', required=True, type=str, help='algorithm')
parser.add_argument('--frac', required=False, default=1.0, type=float, help='number of fractions')
parser.add_argument('--scale', action=argparse.BooleanOptionalAction, help='scale the data before learning')
parser.add_argument('--reduced', action=argparse.BooleanOptionalAction,
                    help='use only "CPUs" and "OVERALL_SIZE" features')

if __name__ == "__main__":
    args = parser.parse_args()
    logger.info(args)
    app_id = app_name_to_id.get(args.app_name, None)

    if app_id is None:
        raise ValueError(f'missing app "{args.app_name}" from app map={str(app_name_to_id)}')

    results_filepath = join(ROOT_DIR, '..', 'execution_results/results.csv')
    results_test_filepath = os.path.join(ROOT_DIR, '..', 'execution_results/results_test.csv')
    results_train_filepath = os.path.join(ROOT_DIR, '..', 'execution_results/results_train.csv')
    df, df_err = get_data_frame(results_filepath, app_id)
    df_test, df_test_err = get_data_frame(results_test_filepath, app_id)
    df_train, df_train_err = get_data_frame(results_train_filepath, app_id)

    if df_err is not None or df_test_err is not None or df_train_err is not None:
        raise ValueError(f'data frame load err')

    if args.reduced:
        x = df[DataFrameColumns.CPUS, DataFrameColumns.OVERALL_SIZE]
        x_test = df_test[DataFrameColumns.CPUS, DataFrameColumns.OVERALL_SIZE]
        x_train = df_train[DataFrameColumns.CPUS, DataFrameColumns.OVERALL_SIZE]
    else:
        x = df.loc[:, df.columns != DataFrameColumns.EXECUTION_TIME]
        x_test = df_test.loc[:, df_test.columns != DataFrameColumns.EXECUTION_TIME]
        x_train = df_train.loc[:, df_train.columns != DataFrameColumns.EXECUTION_TIME]

    y = df.loc[:, df.columns == DataFrameColumns.EXECUTION_TIME]
    y_test = df_test.loc[:, df_test.columns == DataFrameColumns.EXECUTION_TIME]
    y_train = df_train.loc[:, df_train.columns == DataFrameColumns.EXECUTION_TIME]

    if args.scale:
        init_scale(x, y)

    x_test_scaled = transform_x(x_test)
    x_scaled = transform_x(x)
    x_train_scaled = transform_x(x_train)
    y_train_scaled = transform_y(y_train)
    scaled_x_train_df = pd.DataFrame(x_train_scaled, columns=x.columns)
    scaled_y_train_df = pd.DataFrame(y_train_scaled, columns=y.columns)
    x_plot_train = x_train[DataFrameColumns.OVERALL_SIZE]
    y_plot_train = x_train[DataFrameColumns.CPUS]
    z_plot_train = y_train[DataFrameColumns.EXECUTION_TIME]
    x_plot_test = x_test[DataFrameColumns.OVERALL_SIZE]
    y_plot_test = x_test[DataFrameColumns.CPUS]
    z_plot_test = y_test[DataFrameColumns.EXECUTION_TIME]
    # plot data points
    ax = plt.axes(projection='3d')
    ax.set_xlabel('total size [B]', linespacing=10)
    ax.xaxis._axinfo['label']['space_factor'] = 10
    ax.set_ylabel('mCPUSs')
    ax.set_zlabel('time [s]')
    ax.dist = 8
    ax.scatter(x_plot_train, y_plot_train, z_plot_train, c='#2ca02c', alpha=1, label='training points')
    ax.scatter(x_plot_test, y_plot_test, z_plot_test, label='test points', c='#cc0000', alpha=1)
    # Load model
    model_details = ModelDetails(args.app_name, args.frac, args.scale, args.reduced)
    model_filepath, err = get_model_filepath(args.alg, model_details)

    if err is not None:
        raise ValueError(err)

    model = joblib.load(model_filepath)
    z_svr = model.predict(x_scaled)
    z_svr_test = model.predict(x_test_scaled)
    # ML end
    z_svr_test_inverse = inverse_transform_y(z_svr_test)
    y_test_list = list(y_test[DataFrameColumns.EXECUTION_TIME])
    y_train_list = list(y_train[DataFrameColumns.EXECUTION_TIME])
    errors_rel = []
    errors = []

    for index, z_pred in enumerate(z_svr_test_inverse):
        z_pred = z_pred if z_pred > 0 else min(y_train_list)
        z_origin = y_test_list[index]
        error = abs(z_pred - z_origin)
        errors.append(error)
        error_rel = error * 100.0 / z_origin
        errors_rel.append(error_rel)

        if getenv("DEBUG") == "true":
            logger.info('pred: %s' % z_pred)
            logger.info('origin: %s' % z_origin)
            logger.info('error [s] = %s' % error)
            logger.info('error relative [percentage] = %s' % error_rel)

    logger.info('############### SUMMARY ##################')
    logger.info('training set length: %s' % len(y_train_list))
    logger.info('test set length: %s' % len(x_test))
    logger.info('avg time [s] = %s' % str(sum(y_test_list) / len(y_test_list)))
    logger.info('avg error [s] = %s' % str(sum(errors) / len(errors)))
    logger.info('avg error relative [percentage] = %s' % str(sum(errors_rel) / len(errors_rel)))
    # Plot prediction surface
    z_svr_inverse = inverse_transform_y(z_svr)
    x_plot = x[DataFrameColumns.OVERALL_SIZE].to_numpy()
    y_plot = x[DataFrameColumns.CPUS].to_numpy()
    ax.plot_trisurf(x_plot, y_plot, z_svr_inverse, alpha=0.5)
    plt.margins()
    plt.gcf().autofmt_xdate()
    ax.legend()
    plt.title(f'Regression surface using {str(args.alg).upper()} algorithm')
    '''
    model_scheme = ModelDetails(args.app_name, 1.0, args.scale, args.reduced)
    fig_path = os.path.join(ROOT_DIR, 'models', args.alg, 'figures', get_model_name(model_scheme) + '_surf.png')
    plt.savefig(fig_path)
    '''
    plt.show()
