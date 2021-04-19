from os.path import isfile
from typing import Union, Tuple, List, Optional

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class DataFrameColumns:
    APP_ID = 'app_id'
    CPUS = 'cpus'
    OVERALL_SIZE = 'overall_size'
    PARTS = 'parts'
    ELEMENT_AVG_SIZE = 'element_avg_size'
    ELEMENT_MAX_SIZE = 'element_max_size'
    EXECUTION_TIME = 'execution_time'


FEATURE_NAMES = [
    DataFrameColumns.CPUS,
    DataFrameColumns.OVERALL_SIZE,
    DataFrameColumns.PARTS,
    DataFrameColumns.ELEMENT_AVG_SIZE,
    DataFrameColumns.ELEMENT_MAX_SIZE,
]


def get_data_frame(results_filepath: str, app_id: Union[int, None] = None, random_state: int = 0) \
        -> Tuple[Union[None, pd.DataFrame], Union[None, ValueError]]:
    if not isfile(results_filepath):
        return None, ValueError(f'"{results_filepath}" is not a file')

    try:
        df = pd.read_csv(results_filepath, delimiter=',', header=0, names=[
            DataFrameColumns.APP_ID,
            DataFrameColumns.CPUS,
            DataFrameColumns.OVERALL_SIZE,
            DataFrameColumns.PARTS,
            DataFrameColumns.ELEMENT_AVG_SIZE,
            DataFrameColumns.ELEMENT_MAX_SIZE,
            DataFrameColumns.EXECUTION_TIME,
        ])

        if app_id is not None:
            app_df = df.loc[df[DataFrameColumns.APP_ID] == app_id]
            df = app_df.loc[:, df.columns != DataFrameColumns.APP_ID]

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
