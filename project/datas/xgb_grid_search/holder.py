import os
from typing import Tuple, Union
from os.path import isfile, getsize

import pandas as pd
from numpy import ndarray

from project.datas.details import DataDetails
from project.datas.interface.holder import HolderInterface


class XGBData:
    FEATURE_PREFIX = 'f_'
    TARGET_COLUMN = 'target'


class Holder(HolderInterface):
    def __init__(self, data_dir: str) -> None:
        """Load data from given directory."""
        super().__init__(data_dir)

    def get(self) -> Tuple[Union[Tuple[pd.DataFrame, ndarray], None], Union[ValueError, None]]:
        if not isfile(self._data_dir):
            return None, ValueError(f'"{self._data_dir}" is not a file')

        try:
            df = pd.read_csv(self._data_dir, header=None)
            columns = [XGBData.FEATURE_PREFIX + str(i) for i in range(len(df.columns) - 1)]
            columns.append(XGBData.TARGET_COLUMN)
            df.columns = columns
            y = df[XGBData.TARGET_COLUMN].values
            x = df.drop([XGBData.TARGET_COLUMN], axis=1)
            return (x, y), None
        except BaseException as exception:
            return None, ValueError(f'"{self._data_dir}" is not a proper data frame: {str(exception)}')

    def get_details(self) -> DataDetails:
        data_frame_size = 0
        parts = 1,

        if isfile(self._data_dir):
            data_frame_size = getsize(self._data_dir)

        data, get_err = self.get()

        if get_err is None:
            parts = len(data[0].columns)

        return DataDetails(
            overall_size=data_frame_size,
            parts=parts,
            element_avg_size=int(data_frame_size/parts),
            element_max_size=int(data_frame_size/parts),
        )


if __name__ == '__main__':
    holder = Holder(os.path.join('sets', 'ecoli_half.csv'))
    holder.get_details()
