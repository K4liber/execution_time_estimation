# statistic_under_AI
Project for the classes 'Statistic under AI and its application to engineering sciences'.

### Generate data
1. Build app execution image:  
    ```
    sudo docker build -t app_execute --build-arg app_name=video_splitter -f Dockerfile_execute .
    ```
2. Run image:
    ```
   sudo docker run --mount type=bind,source="$(pwd)"/execution_results,target=/app/project/execution_results --cpus=1.1 -e cpus=1.1 app_execute
   ```
