from typing import Dict, Any, List

from sklearn.svm import SVR

from project.models.interface import Algorithm
from project.utils.logger import logger


class AlgSVR(Algorithm):
    @staticmethod
    def get_params_grid() -> Dict[str, List[float]]:
        gamma_min = 0.0001  # rbf kernel influence range
        epsilon_min = 0.000001  # cost range
        c_min = 1000.0  # error cost
        grid = {
            "gamma": [gamma_min * 2 ** x for x in range(8)],
            "epsilon": [epsilon_min * 2 ** x for x in range(7)],
            "C": [c_min * 2 ** x for x in range(11)],
        }
        logger.info(grid)
        return grid

    @staticmethod
    def get() -> Any:
        return SVR(kernel='rbf')
