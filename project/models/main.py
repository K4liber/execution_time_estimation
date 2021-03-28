import argparse
from os.path import join
import sys

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR

sys.path.append('.')

from project.utils.app_ids import app_name_to_id
from project.utils.logger import logger
from project.definitions import ROOT_DIR
from project.models.data import (
    get_data_frame,
    get_training_test_split, DataFrameColumns,
)

parser = argparse.ArgumentParser(description='Model training and tests.')
parser.add_argument('--app_name', required=True, type=str, help='app name')


if __name__ == "__main__":
    args = parser.parse_args()
    logger.info(args)
    app_id = app_name_to_id.get(args.app_name, None)

    if app_id is None:
        raise ValueError(f'missing app "{args.app_name}" from app map={str(app_name_to_id)}')

    results_filepath = join(ROOT_DIR, '..', 'execution_results/results.csv')
    df, df_err = get_data_frame(results_filepath, app_id)

    if df_err is not None:
        raise ValueError(f'data frame load err: {str(df_err)}')

    x, y, x_train, x_test, y_train, y_test = get_training_test_split(df)
    scale_x = StandardScaler().fit(x)
    scale_y = StandardScaler().fit(y)
    x_test_scaled = scale_x.transform(x_test)
    x_scaled = scale_x.transform(x)
    x_train_scaled = scale_x.transform(x_train)
    y_train_scaled = scale_y.transform(y_train)
    scaled_x_train_df = pd.DataFrame(x_train_scaled, columns=x.columns)
    scaled_y_train_df = pd.DataFrame(y_train_scaled, columns=y.columns)
    x_plot_train = x_train[DataFrameColumns.OVERALL_SIZE]
    y_plot_train = x_train[DataFrameColumns.CPUS]
    z_plot_train = y_train[DataFrameColumns.EXECUTION_TIME]
    x_plot_test = x_test[DataFrameColumns.OVERALL_SIZE]
    y_plot_test = x_test[DataFrameColumns.CPUS]
    z_scaled = scaled_y_train_df[DataFrameColumns.EXECUTION_TIME]
    # plot
    ax = plt.axes(projection='3d')
    ax.set_xlabel('total size [B]')
    ax.set_ylabel('cpus')
    ax.set_zlabel('time [s]')
    ax.scatter(x_plot_train, y_plot_train, z_plot_train, c='#2ca02c', alpha=1, label='training points')
    # ML
    svr = GridSearchCV(SVR(kernel='rbf'),
                       param_grid={
                           "gamma": [0.1],
                           "epsilon": [0.001],
                           "C": [300],
                       })
    svr.fit(x_train_scaled, y_train_scaled)
    z_svr = svr.predict(x_scaled)
    z_svr_test = svr.predict(x_test_scaled)
    z_svr_test_inverse = scale_y.inverse_transform(z_svr_test)
    z_plot_test = y_test[DataFrameColumns.EXECUTION_TIME]
    ax.scatter(x_plot_test, y_plot_test, z_plot_test, label='test points', c='#cc0000', alpha=1)
    y_test_list = list(y_test[DataFrameColumns.EXECUTION_TIME])
    y_train_list = list(y_train)
    errors_rel = []
    errors = []

    for index, z_pred in enumerate(z_svr_test_inverse):
        z_pred = z_pred if z_pred > 0 else min(y_train_list)
        print('pred: %s' % z_pred)
        z_origin = y_test_list[index]
        print('origin: %s' % z_origin)
        error = abs(z_pred - z_origin)
        errors.append(error)
        print('error [s] = %s' % error)
        error_rel = error * 100.0 / z_origin
        errors_rel.append(error_rel)
        print('error relative [percentage] = %s' % error_rel)

    print('###############SUMMARY##################')
    print('training set length: %s' % len(y_test_list))
    print('avg time [s] = %s' % str(sum(y_test_list) / len(y_test_list)))
    print('avg error [s] = %s' % str(sum(errors) / len(errors)))
    print('avg error relative [percentage] = %s' % str(sum(errors_rel) / len(errors_rel)))
    z_svr_inverse = scale_y.inverse_transform(z_svr)
    x_plot = x[DataFrameColumns.OVERALL_SIZE].to_numpy()
    y_plot = x[DataFrameColumns.CPUS].to_numpy()
    ax.plot_trisurf(x_plot, y_plot, z_svr_inverse, alpha=0.5)
    # ML end
    plt.margins(tight=False)
    plt.gcf().autofmt_xdate()
    ax.legend()
    plt.show()

