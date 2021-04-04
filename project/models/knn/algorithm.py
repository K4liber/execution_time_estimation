from typing import Dict, Any, List

from sklearn.neighbors import KNeighborsRegressor

from project.models.interface import Algorithm


class AlgKNN(Algorithm):
    @staticmethod
    def get_params_grid() -> Dict[str, List[float]]:
        return {
            'n_neighbors': [1 + x for x in range(15)],
            'weights': ['uniform', 'distance'],
            'p': [1, 2],
        }

    @staticmethod
    def get() -> Any:
        return KNeighborsRegressor()
