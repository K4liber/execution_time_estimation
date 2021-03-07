import argparse
from os import listdir
from os.path import join
import sys
sys.path.append('.')

from project.apps.app.execution import Execution
from project.datas.data.holder import Holder
from project.definitions import ROOT_DIR
from project.utils.logger import logger

parser = argparse.ArgumentParser(description='Execute an app with given data.')
parser.add_argument('--app_name', required=True, type=str, help='app name')
parser.add_argument('--cpus', required=True, type=float, help='cpus fraction')


if __name__ == "__main__":
    args = parser.parse_args()
    logger.info(args)
    results_filepath = join(ROOT_DIR, 'execution_results/results.csv')

    with open(results_filepath, 'a') as results_file:
        sets_dir = join(ROOT_DIR, 'datas/data/sets')

        for filename in listdir(sets_dir):
            logger.info(filename)
            file_dir = join(ROOT_DIR, 'datas/data/sets', filename)
            data_holder = Holder(file_dir)
            data_details = data_holder.get_details()
            logger.info(data_details)
            app_execution = Execution(data_holder)
            execution_time, execution_err = app_execution.run()

            if execution_err is not None:
                logger.error(execution_err)
            else:
                logger.info(f'execution time: {execution_time}')
                data_line = f'{app_execution.id()},{args.cpus},{data_details.overall_size},' \
                            f'{data_details.parts},{data_details.element_avg_size},' \
                            f'{data_details.element_max_size},' + '%.3f' % (execution_time.total_seconds()) + '\n'
                results_file.write(data_line)
