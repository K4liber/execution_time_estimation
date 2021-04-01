from multiprocessing import Pool
from typing import Tuple, Union
from datetime import datetime, timedelta
from os.path import join

import cv2

from project.apps.interface.execution import ExecutionInterface
from project.datas.data.holder import Holder
from project.definitions import ROOT_DIR
from project.utils.app_ids import AppID
from project.utils.logger import logger


class Execution(ExecutionInterface):
    def __init__(self, data: Holder) -> None:
        super().__init__(data)

    def run(self, cpus: float) -> Tuple[timedelta, Union[ValueError, None]]:
        start = datetime.now()

        try:
            video, get_err = self.data.get()

            if get_err is not None:
                return datetime.now() - start, get_err

            pool_size = int(cpus + 0.999)
            logger.info(f'running with pool size = {pool_size}')

            # Find OpenCV version
            (major_ver, minor_ver, subminor_ver) = cv2.__version__.split('.')

            if int(major_ver) < 3:
                fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
                logger.info("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
            else:
                fps = video.get(cv2.CAP_PROP_FPS)
                logger.info("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

            fps_int = int(fps + 0.999)

            with Pool(pool_size) as pool:
                success, image = video.read()
                count = 0
                frame = 0

                while success:
                    if frame == fps_int:
                        frame = 0

                    alphabetic_count = "".join(["0" for _ in range(4-len(str(count)))]) + str(count)
                    name = f'{alphabetic_count}_{frame}.jpg'
                    filepath = join(ROOT_DIR, 'execution_results', 'app_output', name)
                    pool.apply(cv2.imwrite, args=(filepath, image))
                    logger.info(f'image "{filepath}" saved successfully')
                    success, image = video.read()
                    count += 1
                    frame += 1

            return datetime.now() - start, None
        except BaseException as exception:
            return datetime.now() - start, ValueError(exception)

    @classmethod
    def id(cls):
        return AppID.VideoSplitter
