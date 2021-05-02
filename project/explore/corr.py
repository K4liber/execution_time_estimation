import sys
from os.path import join

import seaborn as sns
from matplotlib import pyplot as plt

sys.path.append('.')

from project.explore.utils import get_color_map
from project.models.data import get_data_frame
from project.utils.app_ids import app_name_to_id, AppID, get_app_id_to_name
from project.definitions import ROOT_DIR

from project.utils.logger import logger

if __name__ == '__main__':
    logger.info('pearson correlation')
    results_filepath = join(ROOT_DIR, '..', 'execution_results/results.csv')
    figure_filepath = join(ROOT_DIR, 'explore/figures', 'corr.png')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True)
    # fig.suptitle('Comparing the learning curves per module')
    app_id_to_axes = {
        AppID.XGBoostGridSearch: ax1,
        AppID.ImagesMerger: ax2,
        AppID.VideoSplitter: ax3,
        AppID.FaceRecogniser: ax4
    }
    app_id_to_name = get_app_id_to_name()

    for index, app_id in enumerate(app_name_to_id.values()):
        df, df_err = get_data_frame(results_filepath, app_id, 0, app_id_left=False, short_names=True)
        cor = df.corr(method='spearman')
        app_id_to_axes[app_id].set_title(app_id_to_name[app_id])
        sns.heatmap(cor, cbar=index % 2 == 1, ax=app_id_to_axes[app_id],
                    vmin=-1, vmax=1, annot=False, cmap=get_color_map())

    plt.savefig(figure_filepath, bbox_inches='tight', pad_inches=0)
