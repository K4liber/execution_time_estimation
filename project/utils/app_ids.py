from typing import Dict


class AppID:
    VideoSplitter = 1
    FaceRecogniser = 2
    XGBoostGridSearch = 3
    ImagesMerger = 4


app_name_to_id = {
    'video_splitter': AppID.VideoSplitter,
    'face_recogniser': AppID.FaceRecogniser,
    'xgb_grid_search': AppID.XGBoostGridSearch,
    'images_merger': AppID.ImagesMerger,
}


def get_app_id_to_name() -> Dict[int, str]:
    return {
        app_id: app_name for app_name, app_id in app_name_to_id.items()
    }
