from os.path import join
from unittest import TestCase

from project.definitions import ROOT_DIR
from project.models.data import get_data_frame, DataFrameColumns


class TestDataFrame(TestCase):
    def test_get_data_frame(self):
        results_filepath = join(ROOT_DIR, '..', 'execution_results/test.csv')
        df, df_err = get_data_frame(results_filepath, header=None)
        self.assertEqual(df_err, None, f'get_data_frame err: {str(df_err)}')

    def test_get_data_frame_for_specific_app(self):
        app_id = 1
        results_filepath = join(ROOT_DIR, '..', 'execution_results/test.csv')
        df, df_err = get_data_frame(results_filepath, app_id, header=None)
        self.assertEqual(df_err, None, f'get_data_frame err: {str(df_err)}')
        self.assertEqual(len(df), 21)
