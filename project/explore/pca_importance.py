import sys
from os.path import join
from typing import Tuple

import numpy as np
import pandas as pd
from matplotlib.axes import SubplotBase
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

sys.path.append('.')

from project.models.data import get_data_frame, DataFrameColumns, SHORT_NAME
from project.utils.app_ids import app_name_to_id, AppID, get_app_id_to_name
from project.definitions import ROOT_DIR

from project.utils.logger import logger


def features_impact_bar(model_pca: PCA, axis: SubplotBase, col_names: Tuple[str] = None):
    components = model_pca.components_
    pca_importance = model_pca.explained_variance_ratio_

    if col_names is not None and len(components) != len(col_names):
        raise ValueError(
            f'len(col_names) = {len(col_names)} should be equal to the number of components = {len(components)}')

    normalized_components = []
    previous_bars = [0. for _ in range(x.shape[1])]

    for index, importance in enumerate(pca_importance):
        components_line = abs(components[index])
        components_line_normed = components_line / sum(components_line)
        normalized_importance = np.array(components_line_normed * importance)
        normalized_components.append(normalized_importance)

    normalized_components_t = np.array(normalized_components).transpose()

    for index, normalized_component_t in enumerate(normalized_components_t):
        axis.bar(labels, normalized_component_t, 0.3, bottom=previous_bars,
                 label=col_names[index] if col_names else f'f{index+1}')
        previous_bars = previous_bars + normalized_component_t


if __name__ == '__main__':
    logger.info('pca importance')
    results_filepath = join(ROOT_DIR, '..', 'execution_results/results.csv')
    figure_filepath = join(ROOT_DIR, 'explore/figures', 'pca_importance.png')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True)
    # fig.suptitle('Comparing the learning curves per module')
    app_id_to_axes = {
        AppID.XGBoostGridSearch: ax1,
        AppID.ImagesMerger: ax2,
        AppID.VideoSplitter: ax3,
        AppID.FaceRecogniser: ax4
    }
    app_id_to_name = get_app_id_to_name()

    for app_id in app_name_to_id.values():
        df, df_err = get_data_frame(results_filepath, app_id, 0, app_id_left=False, short_names=True)
        x = df.loc[:, df.columns != SHORT_NAME[DataFrameColumns.EXECUTION_TIME]]
        labels = [f'c{i + 1}' for i in range(x.shape[1])]
        ax = app_id_to_axes[app_id]
        # Initialize
        model_pca = PCA()
        # In general a good idea is to scale the data
        scaler = StandardScaler()
        scaler.fit(x)
        x = scaler.transform(x)
        # Fit transform
        model_pca.fit_transform(x)
        plot_features_impact(
            model_pca, ax,
            tuple([str(col) for col in df.columns if col != SHORT_NAME[DataFrameColumns.EXECUTION_TIME]]))
        ax.set_title(app_id_to_name[app_id])
        ax.legend()

    plt.savefig(figure_filepath, bbox_inches='tight', pad_inches=0)
