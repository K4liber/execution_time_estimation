from os.path import isfile, join
from typing import Union, Tuple, List, Optional
import sys

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.append('.')

from project.definitions import ROOT_DIR
from project.utils.app_ids import app_name_to_id


class DataFrameColumns:
    APP_ID = 'app_id'
    CPUS = 'cpus'
    OVERALL_SIZE = 'overall_size'
    PARTS = 'parts'
    ELEMENT_AVG_SIZE = 'element_avg_size'
    ELEMENT_MAX_SIZE = 'element_max_size'
    EXECUTION_TIME = 'execution_time'


SHORT_NAME = {
    DataFrameColumns.APP_ID: 'APP',
    DataFrameColumns.CPUS: 'CPUS',
    DataFrameColumns.OVERALL_SIZE: 'OVER',
    DataFrameColumns.PARTS: 'PART',
    DataFrameColumns.ELEMENT_AVG_SIZE: 'AVG',
    DataFrameColumns.ELEMENT_MAX_SIZE: 'MAX',
    DataFrameColumns.EXECUTION_TIME: 'TIME',
}


class Const:
    TRAINING_SAMPLES = 60


FEATURE_NAMES = [
    DataFrameColumns.CPUS,
    DataFrameColumns.OVERALL_SIZE,
    DataFrameColumns.PARTS,
    DataFrameColumns.ELEMENT_AVG_SIZE,
    DataFrameColumns.ELEMENT_MAX_SIZE,
]
REDUCED_FEATURES = [
    DataFrameColumns.CPUS,
    DataFrameColumns.OVERALL_SIZE
]


def get_x_y(results_filepath: str, app_id: Union[int, None] = None, reduced: bool = False) -> Tuple[str, str]:
    df, df_err = get_data_frame(results_filepath, app_id)

    if df_err is not None:
        raise ValueError(f'data frame load err: {df_err}')

    if reduced:
        x = df[REDUCED_FEATURES]
    else:
        x = df.loc[:, df.columns != DataFrameColumns.EXECUTION_TIME]

    return x, df.loc[:, df.columns == DataFrameColumns.EXECUTION_TIME]


def get_data_frame(results_filepath: str, app_id: Union[int, None] = None,
                   random_state: int = 0, app_id_left: bool = False, short_names: bool = False,
                   header: Optional[int] = 0) \
                   -> Tuple[Union[None, pd.DataFrame], Union[None, ValueError]]:
    if not isfile(results_filepath):
        return None, ValueError(f'"{results_filepath}" is not a file')

    try:
        df = pd.read_csv(results_filepath, delimiter=',', header=header, names=[
            DataFrameColumns.APP_ID if not short_names else SHORT_NAME[DataFrameColumns.APP_ID],
            DataFrameColumns.CPUS if not short_names else SHORT_NAME[DataFrameColumns.CPUS],
            DataFrameColumns.OVERALL_SIZE if not short_names else SHORT_NAME[DataFrameColumns.OVERALL_SIZE],
            DataFrameColumns.PARTS if not short_names else SHORT_NAME[DataFrameColumns.PARTS],
            DataFrameColumns.ELEMENT_AVG_SIZE if not short_names else SHORT_NAME[DataFrameColumns.ELEMENT_AVG_SIZE],
            DataFrameColumns.ELEMENT_MAX_SIZE if not short_names else SHORT_NAME[DataFrameColumns.ELEMENT_MAX_SIZE],
            DataFrameColumns.EXECUTION_TIME if not short_names else SHORT_NAME[DataFrameColumns.EXECUTION_TIME],
        ])
        app_id_col_name = DataFrameColumns.APP_ID if not short_names else SHORT_NAME[DataFrameColumns.APP_ID]

        if app_id is not None:
            df = df.loc[df[app_id_col_name] == app_id]

        if not app_id_left:
            df = df.loc[:, df.columns != app_id_col_name]

        return df.sample(frac=1, random_state=random_state), None
    except BaseException as exception:
        return None, ValueError(exception)


def get_training_test_split(df: pd.DataFrame, train_fraction: float = 1.0, columns: Union[List[str], None] = None) -> \
        (Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]], Optional[ValueError]):
    if train_fraction < 0.1 or train_fraction > 1.0:
        return None, ValueError(f'"fraction" value should be between 0.1 and 1.0, got = {train_fraction}')

    if columns is None:
        x = df.loc[:, df.columns != DataFrameColumns.EXECUTION_TIME]
    else:
        x = df[columns]

    y = df.loc[:, df.columns == DataFrameColumns.EXECUTION_TIME]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=10)
    x_train = x_train[:int(len(x_train) * train_fraction)]
    y_train = y_train[:int(len(y_train) * train_fraction)]
    return x, y, x_train, x_test, y_train, y_test


if __name__ == '__main__':
    _results_filepath = join(ROOT_DIR, '..', 'execution_results/results.csv')

    for index, _app_id in enumerate(app_name_to_id.values()):
        df, df_err = get_data_frame(_results_filepath, _app_id, 0, True)

        if index == 0:
            df_train = df.iloc[:Const.TRAINING_SAMPLES, :]
            df_test = df.iloc[Const.TRAINING_SAMPLES:, :]
        else:
            df_train = df_train.append(df.iloc[:Const.TRAINING_SAMPLES, :])
            df_test = df_test.append(df.iloc[Const.TRAINING_SAMPLES:, :])

    df_train.to_csv(join(ROOT_DIR, '..', 'execution_results/results_train.csv'), index=False)
    df_test.to_csv(join(ROOT_DIR, '..', 'execution_results/results_test.csv'), index=False)
