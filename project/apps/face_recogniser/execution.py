import os
from multiprocessing import Pool
from os.path import join
from typing import Tuple, Union, List
from datetime import datetime, timedelta

import face_recognition
import matplotlib
from matplotlib import pyplot, patches
import numpy as np
from PIL import Image

from project.apps.interface.execution import ExecutionInterface
from project.datas.data.holder import Holder
from project.definitions import ROOT_DIR
from project.utils.app_ids import AppID
from project.utils.logger import logger


def split_list(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def process(file_paths: List[str]) -> str:
    faced_file_paths = []

    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        faced_file_path = join(ROOT_DIR, 'execution_results', 'app_output', file_name)
        # Mark faces and save the image
        image = np.array(Image.open(file_path))
        #im = Image.fromarray(image)
        #im.save(file_path)
        height: int = image.shape[0]
        width: int = image.shape[1]
        dpi: int = 100
        faces_coords: List[Tuple[int]] = face_recognition.face_locations(image)
        figure = pyplot.figure(frameon=False, dpi=dpi)
        figure.set_size_inches(width / dpi, height / dpi)
        ax = pyplot.Axes(figure, [0., 0., 1., 1.])
        ax.set_axis_off()
        figure.add_axes(ax)
        ax.imshow(image)
        logger.info('adding ' + str(len(faces_coords)) + ' faces to image "' + file_name + '"')
        #fig = pyplot.gcf()
        #fig.savefig(fname=file_path, dpi=dpi, bbox_inches='tight')

        for index in range(len(faces_coords)):
            x_start = faces_coords[index][3]
            y_start = faces_coords[index][0]
            x_width = (faces_coords[index][1] - faces_coords[index][3])
            y_height = (faces_coords[index][2] - faces_coords[index][0])
            rect = patches.Rectangle((x_start, y_start), x_width, y_height,
                                     edgecolor='r', facecolor="none")
            ax.add_patch(rect)

        pyplot.savefig(fname=faced_file_path, dpi=dpi, bbox_inches='tight')
        pyplot.close()
        faced_file_paths.append(faced_file_path)

    return faced_file_paths


class Execution(ExecutionInterface):
    def __init__(self, data: Holder) -> None:
        super().__init__(data)

    def run(self, cpus: float) -> Tuple[timedelta, Union[ValueError, None]]:
        start = datetime.now()

        try:
            image_paths, err = self.data.get()

            if err is not None:
                return datetime.now() - start, err

            pool_size = int(cpus + 0.999)
            logger.info(f'running with pool size = {pool_size}')

            with Pool(pool_size) as pool:
                pool.starmap(
                    process, [(image_paths_i,) for image_paths_i in split_list(image_paths, pool_size)])
                pool.close()
                pool.join()

            execution_time = datetime.now() - start

            return execution_time, None
        except BaseException as exception:
            return datetime.now() - start, ValueError(exception)

    @classmethod
    def id(cls):
        return AppID.FaceRecogniser
