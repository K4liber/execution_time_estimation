import abc
from datetime import timedelta
from typing import Tuple, Union

from project.datas.data.holder import HolderInterface


class ExecutionInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, '__init__') and
                callable(subclass.__init__) and
                hasattr(subclass, 'run') and
                callable(subclass.run) and
                hasattr(subclass, 'id') and
                callable(subclass.id) or
                NotImplemented)

    @abc.abstractmethod
    def __init__(self, data: HolderInterface):
        """Load the app."""
        self._data = data

    @property
    def data(self):
        return self._data

    @abc.abstractmethod
    def run(self) -> Tuple[timedelta, Union[ValueError, None]]:
        """Execute the application. Returns the execution time and the execution error if there is any."""
        pass

    @classmethod
    @abc.abstractmethod
    def id(cls) -> int:
        """Get the id of an app execution."""
        pass
