import logging
from typing import Dict, List, Any

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from project.models.interface import Algorithm

logger = logging.getLogger(__name__)


class AlgPolynomialRegression(Algorithm):
    @staticmethod
    def get_params_grid() -> Dict[str, List[float]]:
        grid = {
            "poly__degree": [1, 2, 3, 4, 5, 6, 7],  # the maximal degree of the polynomial features
            "poly__interaction_only": [True, False],  # If True, only interaction features for the larger degree are produced
            "poly__include_bias": [True, False]  # If True (default), then include a bias column, the feature in which all
            # polynomial powers are zero (i.e. a column of ones - acts as an intercept term in a linear model).
        }
        logger.info(grid)
        return grid

    @staticmethod
    def get() -> Any:
        return Pipeline([
            ('poly', PolynomialFeatures()),
            ('std_scaler', StandardScaler()),
            ('lin_reg', LinearRegression())
        ])
