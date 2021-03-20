#!/bin/bash
docker build -t "$1" --build-arg app_name="$1" -f Dockerfile_execute .
docker stop "$1"
docker rm "$1"

for VARIABLE in 4.0 3.5 3.0 2.5 2.0 1.5 1.0
do
  docker run --mount type=bind,source="$(pwd)"/execution_results,target=/app/project/execution_results --cpus=$VARIABLE -e cpus=$VARIABLE "$1"
done
