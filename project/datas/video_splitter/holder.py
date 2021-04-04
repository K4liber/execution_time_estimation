from typing import Tuple, Union
from os.path import isfile, getsize
import mimetypes

import cv2
from cv2.cv2 import VideoCapture

from project.datas.details import DataDetails
from project.datas.interface.holder import HolderInterface


class Holder(HolderInterface):
    def __init__(self, data_dir: str) -> None:
        """Load data from given directory."""
        super().__init__(data_dir)

    def get(self) -> Tuple[Union[VideoCapture, None], Union[ValueError, None]]:
        if not isfile(self._data_dir):
            return None, ValueError(f'"{self._data_dir}" is not a file')

        if not mimetypes.guess_type(self._data_dir)[0].startswith('video'):
            return None, ValueError(f'"{self._data_dir}" is not a video')

        return cv2.VideoCapture(self._data_dir), None

    def get_details(self) -> DataDetails:
        video_size = 0
        frames = 1

        if isfile(self._data_dir):
            video_size = getsize(self._data_dir)

        video, get_err = self.get()

        if get_err is None:
            # Find OpenCV version
            (major_ver, minor_ver, subminor_ver) = cv2.__version__.split('.')

            if int(major_ver) < 3:
                frames = video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
            else:
                frames = video.get(cv2.CAP_PROP_FRAME_COUNT)

        return DataDetails(
            overall_size=video_size,
            parts=frames,
            element_avg_size=int(video_size/frames),
            element_max_size=int(video_size/frames),
        )
