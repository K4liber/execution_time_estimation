from os.path import join
import sys

import matplotlib.pyplot as plt

sys.path.append('.')

from project.tests.algorithms.utils import arrowed_spines
from project.definitions import ROOT_DIR


def main():
    figure_filepath = join(ROOT_DIR, 'tests/figures', 'empty_chart.png')
    # Plot signal
    fig, ax = plt.subplots()
    fig.set_size_inches(4.5, 3.5)
    arrowed_spines(ax)
    plt.scatter([16], [1], alpha=1, c='g', s=70, marker="d")
    plt.ylim([-0.1, 2.5])
    plt.xlim([0, 32])
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,
        left=False,  # ticks along the bottom edge are off
        right=False,
        labelbottom=False,
        labeltop=False,
        labelright=False,
        labelleft=False
    )
    plt.savefig(figure_filepath, bbox_inches='tight', pad_inches=0.02)


if __name__ == '__main__':
    main()
