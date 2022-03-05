import argparse
import os
from os.path import join
import sys

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

sys.path.append('.')

from project.models.common import get_errors, get_model_details_for_algorithm, get_color, init_scale_from_train_set
from project.models.details import get_model_filepath, ModelDetails
from project.models.scale import transform_x, inverse_transform_y
from project.utils.app_ids import app_name_to_id
from project.utils.logger import logger
from project.definitions import ROOT_DIR
from project.models.data import (
    get_data_frame,
    DataFrameColumns,
)

parser = argparse.ArgumentParser(description='Model training and validation.')
parser.add_argument('--app_name', required=True, type=str, help='app name')
parser.add_argument('--alg', required=True, type=str, help='algorithm')


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

    x_origin = df.loc[:, df.columns != DataFrameColumns.EXECUTION_TIME]
    x_test = df_test.loc[:, df_test.columns != DataFrameColumns.EXECUTION_TIME]
    x_train = df_train.loc[:, df_train.columns != DataFrameColumns.EXECUTION_TIME]
    y = df.loc[:, df.columns == DataFrameColumns.EXECUTION_TIME]
    y_test = df_test.loc[:, df_test.columns == DataFrameColumns.EXECUTION_TIME]
    y_train = df_train.loc[:, df_train.columns == DataFrameColumns.EXECUTION_TIME]
    x_plot_train = x_train[DataFrameColumns.OVERALL_SIZE]
    y_plot_train = x_train[DataFrameColumns.CPUS]
    z_plot_train = y_train[DataFrameColumns.EXECUTION_TIME]
    x_plot_test = x_test[DataFrameColumns.OVERALL_SIZE]
    y_plot_test = x_test[DataFrameColumns.CPUS]
    z_plot_test = y_test[DataFrameColumns.EXECUTION_TIME]
    # plot data points
    ax = plt.axes(projection='3d')
    ax.set_xlabel('over', linespacing=0.1, labelpad=-12)
    ax.set_ylabel('cpus', linespacing=0.1, labelpad=-12)
    ax.set_zlabel('t', linespacing=0.1, labelpad=-15)
    ax.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,
        left=False,  # ticks along the bottom edge are off
        right=False,
        labelbottom=False,
        labeltop=False,
        labelright=False,
        labelleft=False
    )
    ax.dist = 8
    ax.scatter(x_plot_train, y_plot_train, z_plot_train, c='#2ca02c', alpha=1, label='training points')
    ax.scatter(x_plot_test, y_plot_test, z_plot_test, label='test points', c='#cc0000', alpha=1)
    # Load model details
    model_details = get_model_details_for_algorithm(args.app_name, args.alg)

    if model_details.scale:
        init_scale_from_train_set(model_details, app_id)

    x_test = pd.DataFrame(transform_x(x_test), columns=x_test.columns)
    x = pd.DataFrame(transform_x(x_origin), columns=x_origin.columns)
    # Load model
    model_filepath, err = get_model_filepath(args.alg, model_details)

    if err is not None:
        raise ValueError(err)

    model = joblib.load(model_filepath)
    z_all = model.predict(x)
    # Efficiency
    z_test = model.predict(x_test)
    z_test_inverse = inverse_transform_y(z_test)
    y_test_list = list(y_test[DataFrameColumns.EXECUTION_TIME])
    y_train_list = list(y_train[DataFrameColumns.EXECUTION_TIME])
    errors, errors_rel = get_errors(y_test_list, z_test_inverse)
    logger.info('############### SUMMARY ##################')
    logger.info('avg time [s] = %s' % str(sum(y_test_list) / len(y_test_list)))
    logger.info('avg error [s] = %s' % str(sum(errors) / len(errors)))
    logger.info('avg error relative [percentage] = %s' % str(sum(errors_rel) / len(errors_rel)))
    logger.info(f'best params: {str(model.get_params())}')
    # Plot prediction surface
    z_inverse = inverse_transform_y(z_all)
    x_plot = x_origin[DataFrameColumns.OVERALL_SIZE].to_numpy()
    y_plot = x_origin[DataFrameColumns.CPUS].to_numpy()
    ax.plot_trisurf(x_plot, y_plot, z_inverse, alpha=0.5, color=get_color(args.alg))
    fake_legend_point = matplotlib.lines.Line2D([0], [0], linestyle="solid", c=get_color(args.alg))

    plt.margins()
    plt.gcf().autofmt_xdate()
    handles, labels = ax.get_legend_handles_labels()
    handles.append(fake_legend_point)
    labels.append(args.alg)
    ax.legend(handles, labels, loc='upper left')
    ax.view_init(elev=20., azim=140)
    model_scheme = ModelDetails(args.app_name, 1.0, True, False)
    fig_path = os.path.join(ROOT_DIR, 'models', 'figures', '_'.join([args.alg, args.app_name, 'surf.png']))
    plt.savefig(fig_path, bbox_inches='tight', pad_inches=0)
