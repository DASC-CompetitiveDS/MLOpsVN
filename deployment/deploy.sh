#!/bin/bash

cmd=$1

# constants
IMAGE_NAME="model_predictor"
IMAGE_TAG=$(git describe --always)

if [[ -z "$cmd" ]]; then
    echo "Missing command"
    exit 1
fi

run_predictor() {
    model_config_path1=$1
    model_config_path2=$2
    specific_handle=$3
    port=$4
    predictor_config_path=$5
    mlflow_uri=$6
    
    echo "check $path"
    docker build -f deployment/model_predictor/Dockerfile -t $IMAGE_NAME:$IMAGE_TAG .
    IMAGE_NAME=$IMAGE_NAME IMAGE_TAG=$IMAGE_TAG \
        MODEL_CONFIG_PATH=$model_path PREDICT_PATH=$path PORT=$port PREDICTOR_CONFIG_PATH=$predictor_config_path MLFLOW_URI=$mlflow_uri\
        docker-compose -f deployment/model_predictor/docker-compose.yml -p ${IMAGE_NAME} up -d --remove-orphans
}

shift

case $cmd in
run_predictor)
    run_predictor "$@"
    ;;
*)
    echo -n "Unknown command: $cmd"
    exit 1
    ;;
esac
