#!/bin/sh

export DOCKER_NAME=mmdetection3d
export DOCKER_PORT=8888
export DOCKER_VOLUME=$1

# Run docker container
#docker run -e JUPYTER_TOKEN="password" -p $DOCKER_PORT:8888 -v $DOCKER_VOLUME:/workspace/storage --gpus all $DOCKER_NAME
docker run -e JUPYTER_TOKEN="password" -p $DOCKER_PORT:8888 -v $DOCKER_VOLUME:/mmdetection3d/data --gpus all $DOCKER_NAME