# Estimation of Execution Time for Computing Tasks

The  aim  of  this  work  is  to  estimate  the  execution time  of  data  processing tasks (specific executions of a program or an algorithm) before their execution. 

## Generate data
### Available applications
1. `video_splitter`
2. `face_recogniser`
3. `xgb_grid_search`
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

Dataset consists of following features (the same for each application):  
- overall size
- average size
- maximum size
- number of elements
- cpus

One can reduce the set of features to the two most important ones:
- overall size
- cpus

In order to reduce the features set an environment variable `REDUCED=true`.

### Available algorithms
1. `svr`
2. `xgb`
3. `knn`
4. `pol`

###
1. Install requirements:
   ```
    python -m pip install -r requirements.txt
   ```
1. Grid search:
    ```
    python project/models/grid_search.py --app_name APP_NAME --alg ALGORITHM_NAME --frac 9
    ```
2. Plot learning curve:
   ```
    python project/models/learning_curve.py --app_name APP_NAME --alg ALGORITHM_NAME
   ```
3. Plot model surface:
    ```
    python project/models/plot_surface.py --app_name APP_NAME --alg ALGORITHM_NAME
    ```
