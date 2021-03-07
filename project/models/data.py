from os.path import isfile
from typing import Union, Tuple

import pandas as pd


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
            df = df.loc[df[DataFrameColumns.APP_ID] == app_id]

        return df, None
    except BaseException as exception:
        return None, ValueError(exception)
