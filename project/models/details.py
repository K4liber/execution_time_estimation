import os
from typing import Union

from project.definitions import ROOT_DIR


class ModelDetails:
    def __init__(self, app_name: str, frac: float, scale: bool, reduced: bool):
        self._app_name = app_name
        self._frac = frac
        self._scale = scale
        self._reduced = reduced

    @property
    def app_name(self):
        return self._app_name

    @property
    def frac(self):
        return self._frac

    @property
    def scale(self):
        return self._scale

    @property
    def reduced(self):
        return self._reduced

    def the_same_run(self, comparing_details) -> bool:
        if self.scale == comparing_details.scale and \
                self.reduced == comparing_details.reduced and \
                self.app_name == comparing_details.app_name:
            return True
        else:
            return False


def get_model_details(filename: str) -> (Union[ModelDetails, None], Union[ValueError, None]):
    try:
        filename_split = '.'.join(filename.split('.')[:-1]).split('_')
        app_name = '_'.join(filename_split[:-3])
        frac = float(filename_split[-3])
        scale = filename_split[-2] == '1'
        reduced = filename_split[-1] == '1'
        return ModelDetails(app_name, frac, scale, reduced), None
    except BaseException as err:
        return None, err


def get_model_filepath(alg: str, model_details: ModelDetails) -> (Union[str, None], Union[ValueError, None]):
    model_name = f'{model_details.app_name}_{round(model_details.frac, 1)}'
    model_name = model_name + '_' + ('1' if model_details.scale else '0')
    model_name = model_name + '_' + ('1' if model_details.reduced else '0')
    model_name = model_name + '.pkl'
    model_path = os.path.join(ROOT_DIR, 'models', alg, model_name)

    if not os.path.isfile(model_path):
        return None, ValueError(f'"{model_path}" is not a file')
    else:
        return model_path, None
