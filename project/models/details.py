import os
from dataclasses import dataclass
from typing import Union

from project.definitions import ROOT_DIR


@dataclass
class ModelDetails:
    app_name: str
    frac: float
    scale: bool
    reduced: bool


def get_model_name(model_details: ModelDetails) -> str:
    model_name = f'{model_details.app_name}_{round(model_details.frac, 1)}'
    model_name = model_name + '_' + ('1' if model_details.scale else '0')
    return model_name + '_' + ('1' if model_details.reduced else '0')


def get_model_filepath(alg: str, model_details: ModelDetails) -> (Union[str, None], Union[ValueError, None]):
    model_name = get_model_name(model_details)
    model_name = model_name + '.pkl'
    model_path = os.path.join(ROOT_DIR, 'models', alg, model_name)

    if not os.path.isfile(model_path):
        return None, ValueError(f'"{model_path}" is not a file')
    else:
        return model_path, None
