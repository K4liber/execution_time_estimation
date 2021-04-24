import copy
import json

import matplotlib.colors as plt_colors

from project.utils.logger import logger


def get_color_map():
    color_dict = {'green': ([0.0, 0.0, 0.0],
                          [0.5, 0.0, 0.0],
                          [1.0, 1.0, 1.0]),
                  'blue': ([0.0, 0.0, 0.0],
                           [1.0, 0.0, 0.0]),
                  'red': ([0.0, 0.0, 1.0],
                            [0.5, 0.0, 0.0],
                            [1.0, 0.0, 0.0])}
    logger.info(json.dumps(color_dict, indent=4))
    return plt_colors.LinearSegmentedColormap(
        'my_colormap', color_dict, 100)
