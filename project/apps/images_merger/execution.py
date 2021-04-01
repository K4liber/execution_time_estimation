from typing import Tuple, Union
from datetime import datetime, timedelta
from os.path import join

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
            image_paths, get_err = self.data.get()

            if get_err is not None:
                return datetime.now() - start, get_err

            pool_size = int(cpus + 0.999)
            logger.info(f'running with pool size = {pool_size}')

            import os
            import moviepy.video.io.ImageSequenceClip
            image_paths.sort()

            try:
                fps = max([int(image_path.split('.')[0].split('_')[-1]) for image_path in image_paths]) + 1
            except ValueError as err:
                logger.error(f'error while reading fps from file names: {str(err)}')
                return datetime.now() - start, None

            logger.info(f'saving movie from {str(len(image_paths))} images with fps = {str(fps)}')
            clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_paths, fps=fps)
            output_file_path = join(ROOT_DIR, 'execution_results', 'app_output', f'video.mp4')
            clip.write_videofile(output_file_path)
            return datetime.now() - start, None
        except BaseException as exception:
            return datetime.now() - start, ValueError(exception)

    @classmethod
    def id(cls):
        return AppID.ImagesMerger
