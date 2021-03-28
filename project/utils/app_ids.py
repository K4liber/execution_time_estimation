class AppID:
    VideoSplitter = 1
    FaceRecogniser = 2
    XGBoostGridSearch = 3


app_name_to_id = {
    'video_splitter': AppID.VideoSplitter,
    'face_recogniser': AppID.FaceRecogniser,
    'xgboost_grid_search': AppID.XGBoostGridSearch,
}
