#!/bin/bash
for ALGORITHM in "pol"
do
  for REDUCED in "true" ""
  do
    for APP_NAME in "face_recogniser" "xgb_grid_search" "images_merger" "video_splitter"
    do
      REDUCED=$REDUCED python project/models/grid_search.py --app_name $APP_NAME --alg $ALGORITHM --frac 1
    done
  done
done
