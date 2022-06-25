import logging
import sys
from dataclasses import dataclass
from os.path import join

import pandas as pd

sys.path.append('.')

from project.definitions import ROOT_DIR

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Column:
    ALG = 'algorithm_name'
    APP = 'application_name'
    FRAC = 'fraction'
    TRAINING_TIME = 'training_time'
    EVALUATION_TIME = 'evaluation_time'


_columns = [Column.ALG, Column.APP, Column.FRAC, Column.TRAINING_TIME, Column.EVALUATION_TIME]


def times_summary():
    times_filepath = join(ROOT_DIR, '..', 'execution_results', 'times.csv')
    times_reduced_filepath = join(ROOT_DIR, '..', 'execution_results', 'times_reduced.csv')
    times_df = pd.read_csv(times_filepath)
    times_reduced_df = pd.read_csv(times_reduced_filepath)
    times_per_algorithm_df = times_df.groupby(Column.ALG).mean()
    times_reduced_per_algorithm_df = times_reduced_df.groupby(Column.ALG).mean()

    for algorithm in times_per_algorithm_df.index:
        algorithm_str = str(algorithm)
        training_time = times_per_algorithm_df.loc[algorithm_str][Column.TRAINING_TIME]
        training_time_reduced = times_reduced_per_algorithm_df.loc[algorithm_str][Column.TRAINING_TIME]
        logger.info(f'{algorithm_str.upper()} training time: original: {round(training_time, 2)}s, '
                    f'reduced: {round(training_time_reduced, 2)}s')


if __name__ == '__main__':
    times_summary()
