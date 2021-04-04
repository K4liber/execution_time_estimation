from typing import Dict, Any, List

from xgboost import XGBRegressor

from project.models.interface import Algorithm


class AlgXGB(Algorithm):
    @staticmethod
    def get_params_grid() -> Dict[str, List[float]]:
        return {
            'n_estimators': [5, 10, 20, 40, 100],
            'max_depth': [1, 2, 3, 4, 5, 6, 7],
            'eta': [0.01 * 2 ** x for x in range(8)],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
        }

    @staticmethod
    def get() -> Any:
        return XGBRegressor()
