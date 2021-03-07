from os.path import join
from unittest import TestCase

from project.definitions import ROOT_DIR
from project.models.data import get_data_frame
from project.utils.logger import logger


class TestDataFrame(TestCase):
    def test_get_data_frame(self):
        results_filepath = join(ROOT_DIR, '..', 'execution_results/results.csv')
        df, df_err = get_data_frame(results_filepath)
        self.assertEqual(df_err, None, f'get_data_frame err: {str(df_err)}')
        logger.info(df)
