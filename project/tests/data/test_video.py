import mimetypes
from os import remove
from os.path import join, isfile
from typing import List
from unittest import TestCase

import cv2

from project.data.video import VideoData
from project.definitions import ROOT_DIR


class TestVideoData(TestCase):
    def test_splitting_into_images(self):
        video_data = VideoData(join(ROOT_DIR, 'apps/video_splitter/datasets/01/people_10s_60fps.mp4'))
        video, get_err = video_data.get()
        self.assertEqual(get_err, None, 'get video data error')
        success, image = video.read()
        count = 0
        images_names: List[str] = []

        while success:
            name = f'{count}.jpg'
            cv2.imwrite(name, image)
            images_names.append(name)
            success, image = video.read()
            count += 1

        self.assertEqual(count, 600 + 1, 'the video is 10s long with 60fps speed so it should be (600 + 1) frames')

        for name in images_names:
            try:
                self.assertTrue(isfile(name), f'"{name}" is not a file')
                self.assertTrue(mimetypes.guess_type(name)[0].startswith('image'), f'"{name}" is not an image')
                remove(name)
            except BaseException as exception:
                self.assertEqual(exception, None,
                                 f'exception occurred while testing the output images: {str(exception)}')
