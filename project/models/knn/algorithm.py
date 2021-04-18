from typing import Dict, Any, List

from sklearn.neighbors import KNeighborsRegressor

from project.models.interface import Algorithm
from project.utils.logger import logger


class KNNParam:
    MAX_N = 'max_n'


class AlgKNN(Algorithm):
    @staticmethod
    def get_params_grid(params: dict = None) -> Dict[str, List[float]]:
        max_n = 15

        if params is not None and KNNParam.MAX_N in params:
            max_n = params[KNNParam.MAX_N]

        grid = {
            'n_neighbors': [1 + x for x in range(min(max_n, 15))],
            'weights': ['uniform', 'distance'],
            'p': [1, 2],
        }
        logger.info(grid)
        return grid

    @staticmethod
    def get() -> Any:
        return KNeighborsRegressor()
