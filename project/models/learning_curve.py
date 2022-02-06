import argparse
import os
import sys

import matplotlib.pyplot as plt

sys.path.append('.')

from project.models.details import ModelDetails, get_model_name

from project.definitions import ROOT_DIR
from project.utils.logger import logger
from project.models.common import get_model_details_for_algorithm, plot_app_learning_curve

parser = argparse.ArgumentParser(description='draw a learning curve')
parser.add_argument('--app_name', required=True, type=str, help='app name')
parser.add_argument('--alg', required=True, type=str, help='algorithm')


def learning_curve(application_name: str, algorithm_name: str, model_scheme: ModelDetails):
    ax = plt.subplot()
    plot_app_learning_curve(application_name, algorithm_name, ax)
    plt.xlabel('training samples quantity')
    plt.ylabel('regression relative error [%]')
    plt.title(f'Learning curve ({algorithm_name}, {application_name})')
    ax.label_outer()
    ax.set_ylim(0, 200)
    ax.legend()
    fig_path = os.path.join(ROOT_DIR, 'models', algorithm_name, 'figures', get_model_name(model_scheme) + '.png')
    plt.savefig(fig_path)


if __name__ == '__main__':
    args = parser.parse_args()
    logger.info(args)
    learning_curve(args.app_name, args.alg, get_model_details_for_algorithm(args.app_name, args.alg))
