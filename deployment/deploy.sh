#!/bin/bash

IMAGE_NAME=$1
IMAGE_TAG=$(git describe --always)


run_predictor() {
    model_path=$2
    path=$3
    port=$4
    predictor_config_path=$5
    server=$6
    
    echo "check $path"
    docker build -f deployment/model_predictor/Dockerfile -t $IMAGE_NAME:$IMAGE_TAG .
    IMAGE_NAME=$IMAGE_NAME IMAGE_TAG=$IMAGE_TAG \
        MODEL_CONFIG_PATH=$model_path PREDICT_PATH=$path PORT=$port PREDICTOR_CONFIG_PATH=$predictor_config_path SERVER=$server\
        docker-compose -f deployment/model_predictor/docker-compose.yml -p ${IMAGE_NAME} up -d --remove-orphans
}

run_predictor "$@"