from typing import Any, Tuple, Union

from project.datas.details import DataDetails
from project.datas.interface.holder import HolderInterface


class Holder(HolderInterface):
    def __init__(self, data_dir: str):
        super().__init__(data_dir)

    def get(self) -> Tuple[Any, Union[ValueError, None]]:
        pass

    def get_details(self) -> DataDetails:
        pass
