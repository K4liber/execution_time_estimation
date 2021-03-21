import os
from multiprocessing import Pool
from typing import Tuple, Union
from datetime import datetime, timedelta

import face_recognition
import matplotlib
from matplotlib import pyplot, patches

from project.apps.interface.execution import ExecutionInterface
from project.datas.data.holder import Holder
from project.utils.app_ids import AppID
from project.utils.logger import logger


processed_part: str = '_processed.'


def faces_coords(filepath: str):
    image = face_recognition.load_image_file(filepath)
    return face_recognition.face_locations(image)


def process(file_path: str) -> str:
    file_name = os.path.basename(file_path)
    file_dir = os.path.dirname(file_path)
    faced_file_name = file_name.split('.')[0] + processed_part + file_name.split('.')[1]
    faced_file_path = file_dir + faced_file_name
    coords = faces_coords(file_path)
    img = matplotlib.image.imread(file_path)
    figure, ax = pyplot.subplots(1)
    ax.imshow(img)

    for i in range(len(coords)):
        x_start = coords[i][3]
        y_start = coords[i][0]
        x_width = (coords[i][1] - coords[i][3])
        y_height = (coords[i][2] - coords[i][0])
        rect = patches.Rectangle((x_start, y_start), x_width, y_height,
                                 edgecolor='r', facecolor="none")
        ax.add_patch(rect)

    pyplot.savefig(fname=faced_file_path)
    pyplot.close(fig=figure)
    return faced_file_path


class Execution(ExecutionInterface):
    def __init__(self, data: Holder) -> None:
        super().__init__(data)

    def run(self, cpus: float) -> Tuple[timedelta, Union[ValueError, None]]:
        start = datetime.now()

        try:
            image_paths, err = self.data.get()

            if err is not None:
                return datetime.now() - start, err

            pool_size = len(image_paths)
            logger.info(f'running with pool size = {pool_size}')

            with Pool(pool_size) as pool:
                file_processed_paths = pool.starmap(process, [(image_path,) for image_path in image_paths])
                pool.close()
                pool.join()

            execution_time = datetime.now() - start

            for file_processed_path in file_processed_paths:
                os.remove(file_processed_path)

            return execution_time, None
        except BaseException as exception:
            return datetime.now() - start, ValueError(exception)

    @classmethod
    def id(cls):
        return AppID.FaceRecogniser
