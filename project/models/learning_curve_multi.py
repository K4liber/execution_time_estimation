import os
import sys

import matplotlib.pyplot as plt

sys.path.append('.')

from project.utils.app_ids import app_name_to_id, AppID
from project.definitions import ROOT_DIR
from project.models.common import plot_app_learning_curve


def learning_curve_multi(first_algorithm_name: str, second_algorithm_name: str):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True)
    app_id_to_axes = {
        AppID.XGBoostGridSearch: ax1,
        AppID.ImagesMerger: ax2,
        AppID.VideoSplitter: ax3,
        AppID.FaceRecogniser: ax4
    }

    for app_name, app_id in app_name_to_id.items():
        plot_app_learning_curve(app_name, first_algorithm_name, app_id_to_axes[app_id])
        plot_app_learning_curve(app_name, second_algorithm_name, app_id_to_axes[app_id])

    for ax in fig.get_axes():
        ax.label_outer()
        ax.set_ylim(0, 200)
        ax.legend()

    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel('number of training samples')
    plt.ylabel('regression relative error [%]')
    plt.tight_layout()
    fig_path = os.path.join(ROOT_DIR, 'models', 'figures',
                            f'learning_curve_multi_{first_algorithm_name}_{second_algorithm_name}' + '.png')
    plt.savefig(fig_path, bbox_inches='tight', pad_inches=0.05)


if __name__ == '__main__':
    learning_curve_multi(sys.argv[1], sys.argv[2])
