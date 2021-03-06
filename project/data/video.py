from typing import Tuple, Union
from os.path import isfile
import mimetypes

import cv2
from cv2.cv2 import VideoCapture

from project.data.interface import DataInterface


class VideoData(DataInterface):
    def __init__(self, data_dir: str) -> None:
        """Load data from given directory."""
        super().__init__(data_dir)

    def get(self) -> Tuple[Union[VideoCapture, None], Union[ValueError, None]]:
        if not isfile(self._data_dir):
            return None, ValueError(f'"{self._data_dir}" is not a file')

        if not mimetypes.guess_type(self._data_dir)[0].startswith('video'):
            return None, ValueError(f'"{self._data_dir}" is not a video')

        return cv2.VideoCapture(self._data_dir), None
