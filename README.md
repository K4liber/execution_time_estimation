# statistic_under_AI
Project for the classes 'Statistic under AI and its application to engineering sciences'.
## Generate data
### Available apps
1. `video_splitter`
2. `face_recogniser`
3. `xgboost_grid_search`
4. `images_merger`  
### Auto data generator
1. Build app execution image:  
    ```
    sudo docker build -t APP_NAME --build-arg app_name=APP_NAME -f Dockerfile_execute .
    ```
2. Run image:
    ```
   sudo docker run --mount type=bind,source="$(pwd)"/execution_results,target=/app/project/execution_results --cpus=1.1 -e cpus=1.1 APP_NAME
   ```  

or just run the bash script to automatically generate date for an application using different resources limits:
```
bash app_executions.sh APP_NAME
```
Here you can find the full data set from all apps executions with all available input data sets:
[results.csv](/execution_results/results.csv)
## Train and validate the model
### Available algorithms
1. `svr`
2. `xgb`
3. `knn`
###
1. Install requirements:
   ```
    python -m pip install -r requirements.txt
   ```
1. Grid search:
    ```
    python project/models/grid_search.py --app_name APP_NAME --alg ALGORITHM_NAME --frac 10 <--scale> <--reduced>
    ```
   use `--scale` flag to scale the data  
   use `--reduced` flag to use onlu `mCPUs` and `overall_size` features  
2. Plot learning curve:
   ```
    python project/models/learning_curve.py --app_name APP_NAME --alg ALGORITHM_NAME <--scale> <--reduced>
   ```
3. Plot model surface:
    ```
    python project/models/plot_surface.py --app_name APP_NAME --alg ALGORITHM_NAME <--scale> <--reduced>
    ```
