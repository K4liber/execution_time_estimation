from typing import Tuple, Union
from datetime import datetime, timedelta

from cv2 import imwrite

from project.apps.interface.execution import ExecutionInterface
from project.datas.data.holder import Holder
from project.utils.logger import logger


class Execution(ExecutionInterface):
    def __init__(self, data: Holder) -> None:
        super().__init__(data)

    def run(self) -> Tuple[timedelta, Union[ValueError, None]]:
        # TODO make it work in multiprocessing mode
        start = datetime.now()

        try:
            video, get_err = self.data.get()

            if get_err is not None:
                return datetime.now() - start, get_err

            success, image = video.read()
            count = 0

            while success:
                name = f'{count}.jpg'
                imwrite(name, image)
                logger.info(f'image "{name}" saved successfully')
                success, image = video.read()
                count += 1

            return datetime.now() - start, None
        except BaseException as exception:
            return datetime.now() - start, ValueError(exception)

    @classmethod
    def id(cls):
        return 1
