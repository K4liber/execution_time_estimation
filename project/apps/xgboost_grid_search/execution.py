from typing import Tuple, Union
from datetime import datetime, timedelta

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from xgboost import XGBClassifier
import pandas as pd

from project.apps.interface.execution import ExecutionInterface
from project.datas.data.holder import Holder
from project.utils.app_ids import AppID
from project.utils.logger import logger


class Execution(ExecutionInterface):
    def __init__(self, data: Holder) -> None:
        super().__init__(data)

    def run(self, cpus: float) -> Tuple[timedelta, Union[ValueError, None]]:
        start = datetime.now()
        n_jobs = int(cpus + 0.999)
        logger.info(f'running with n_jobs = {n_jobs}')

        try:
            data, get_err = self.data.get()

            if get_err is not None:
                return datetime.now() - start, get_err

            y = data[1]
            x = data[0]
            # A parameter grid for XGBoost
            params = {
                'min_child_weight': [1, 5, 10],
                'gamma': [0.5, 1, 1.5, 2, 5],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'max_depth': [3, 4, 5]
            }
            xgb = XGBClassifier(learning_rate=0.02, n_estimators=600, objective='binary:logistic',
                                silent=True, nthread=1)
            folds = 3
            param_comb = 5
            skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1001)
            random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, scoring='accuracy',
                                               n_jobs=n_jobs, cv=skf.split(x, y), verbose=3, random_state=1001)
            random_search.fit(x, y)
            logger.info('\n All results:')
            logger.info(random_search.cv_results_)
            logger.info('\n Best estimator:')
            logger.info(random_search.best_estimator_)
            logger.info('\n Best score for %d-fold search with %d parameter combinations:' % (
                folds, param_comb))
            logger.info(random_search.best_score_)
            logger.info('\n Best hyperparameters:')
            logger.info(random_search.best_params_)
            results = pd.DataFrame(random_search.cv_results_)
            results.to_csv('xgb-random-grid-search-results-01.csv', index=False)
            return datetime.now() - start, None
        except BaseException as exception:
            return datetime.now() - start, ValueError(exception)

    @classmethod
    def id(cls):
        return AppID.XGBoostGridSearch
