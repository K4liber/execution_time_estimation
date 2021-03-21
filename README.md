# statistic_under_AI
Project for the classes 'Statistic under AI and its application to engineering sciences'.
### Available apps
1. `video_splitter`
2. `face_recogniser`
### Generate data
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
### Visualize the modeling results
```
python project/models/main.py --app_name APP_NAME
```
