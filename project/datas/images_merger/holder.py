import os
from typing import Tuple, Union, List
from os.path import getsize, isdir
import mimetypes


from project.datas.details import DataDetails
from project.datas.interface.holder import HolderInterface


class Holder(HolderInterface):
    def __init__(self, data_dir: str) -> None:
        """Load data from given directory."""
        super().__init__(data_dir)

    def get(self) -> Tuple[Union[List[str], None], Union[ValueError, None]]:
        if not isdir(self._data_dir):
            return None, ValueError(f'"{self._data_dir}" is not a directory')

        image_paths: List[str] = []

        for photo_name in os.listdir(self._data_dir):
            image_path = os.path.join(self._data_dir, photo_name)

            if os.path.isdir(image_path):
                continue

            if not mimetypes.guess_type(image_path)[0].startswith('image'):
                continue

            image_paths.append(image_path)

        if len(image_paths) == 0:
            return None, ValueError(f'cannot found any image in the directory "{self._data_dir}"')
        else:
            return image_paths, None

    def get_details(self) -> DataDetails:
        image_paths, err = self.get()

        if err is not None:
            return DataDetails(
                overall_size=0,
                parts=0,
                element_avg_size=0,
                element_max_size=0,
            )

        image_sizes: List[int] = []

        for image_path in image_paths:
            image_sizes.append(getsize(image_path))

        return DataDetails(
            overall_size=sum(image_sizes),
            parts=len(image_sizes),
            element_avg_size=int(sum(image_sizes)/len(image_sizes)),
            element_max_size=max(image_sizes),
        )
