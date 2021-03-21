from multiprocessing import Pool
from typing import Tuple, Union
from datetime import datetime, timedelta

from cv2 import imwrite

from project.apps.interface.execution import ExecutionInterface
from project.datas.data.holder import Holder
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

            with Pool(pool_size) as pool:
                success, image = video.read()
                count = 0

                while success:
                    name = f'{count}.jpg'
                    pool.apply_async(imwrite, args=(name, image))
                    logger.info(f'image "{name}" saved successfully')
                    success, image = video.read()
                    count += 1

                return datetime.now() - start, None
        except BaseException as exception:
            return datetime.now() - start, ValueError(exception)

    @classmethod
    def id(cls):
        return AppID.VideoSplitter
