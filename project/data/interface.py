import abc
from typing import Any, Tuple, Union


class DataInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, '__init__') and
                callable(subclass.__init__) or
                hasattr(subclass, 'get') and
                callable(subclass.get) or
                NotImplemented)

    @abc.abstractmethod
    def __init__(self, data_dir: str):
        """Load the data"""
        self._data_dir = data_dir

    @abc.abstractmethod
    def get(self) -> Tuple[Any, Union[ValueError, None]]:
        """Get the data."""
        pass
