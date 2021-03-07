from datetime import timedelta
from typing import Tuple, Union

from project.apps.interface.execution import ExecutionInterface
from project.datas.interface.holder import HolderInterface


class Execution(ExecutionInterface):
    @classmethod
    def id(cls) -> int:
        pass

    def run(self) -> Tuple[timedelta, Union[ValueError, None]]:
        pass

    def __init__(self, data: HolderInterface):
        super().__init__(data)
