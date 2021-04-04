import abc
from typing import Dict, List, Any


class Algorithm(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'get_params_grid') and
                callable(subclass.get_params_grid) and
                hasattr(subclass, 'get') and
                callable(subclass.get) or
                NotImplemented)

    @staticmethod
    @abc.abstractmethod
    def get_params_grid() -> Dict[str, List[float]]:
        pass

    @staticmethod
    @abc.abstractmethod
    def get() -> Any:
        pass
