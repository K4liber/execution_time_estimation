import argparse
import os
import sys

from matplotlib import pyplot as plt
from scipy.stats import ks_2samp

sys.path.append('.')

from project.utils.app_ids import AppID, get_app_id_to_name
from project.definitions import ROOT_DIR
from project.models.common import get_name_suffix

parser = argparse.ArgumentParser(description='Model training and validation.')
parser.add_argument('--alg', required=True, type=str, help='algorithm')

if __name__ == "__main__":
    args = parser.parse_args()
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True)
    app_id_to_name = get_app_id_to_name()
    app_name_to_axes = {
        app_id_to_name[AppID.VideoSplitter]: ax1,
        app_id_to_name[AppID.FaceRecogniser]: ax2,
        app_id_to_name[AppID.XGBoostGridSearch]: ax3,
        app_id_to_name[AppID.ImagesMerger]: ax4,
    }
    results = dict()
    p_values = []

    for app_name, ax in app_name_to_axes.items():
        for is_reduced in {True, False}:
            results_filepath = os.path.join(ROOT_DIR, '..', 'execution_results',
                                            f'{args.alg}_{app_name}_errors{get_name_suffix(is_reduced)}.csv')
            results[is_reduced] = []

            with open(results_filepath, 'r') as results_file:
                line = results_file.readline()

                while line:
                    results[is_reduced].append(float(line))
                    line = results_file.readline()

            ax.hist(results[is_reduced], bins=20, label=str(is_reduced))

        ks = ks_2samp(results[True], results[False])
        ax.legend()
        p_value = 100 * ks.pvalue
        p_values.append(p_value)
        ax.set_title(f'{app_name}, p: {p_value:.3f}%')

    fig.suptitle(f'Algorithm: {args.alg.upper()}, average p-value: {sum(p_values)/len(p_values):.3f}%')
    plt.tight_layout()
    file_name = f'{args.alg}_p_value.png'
    fig_path = os.path.join(ROOT_DIR, 'models', 'figures', file_name)
    plt.savefig(fig_path, bbox_inches='tight', pad_inches=0.05)
