from os.path import isfile
from typing import Union, Tuple

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


def get_data_frame(results_filepath: str, app_id: Union[int, None] = None) -> Tuple[Union[None, pd.DataFrame],
                                                                                    Union[None, ValueError]]:
    if not isfile(results_filepath):
        return None, ValueError(f'"{results_filepath}" is not a file')

    try:
        df = pd.read_csv(results_filepath, delimiter=',', names=[
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

        return df, None
    except BaseException as exception:
        return None, ValueError(exception)


def get_training_test_split(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                       np.ndarray, np.ndarray, np.ndarray]:
    x = df.loc[:, df.columns != DataFrameColumns.EXECUTION_TIME]
    y = df.loc[:, df.columns == DataFrameColumns.EXECUTION_TIME]
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=0.33, random_state=42)
    return x, y, x_train, x_test, y_train, y_test
