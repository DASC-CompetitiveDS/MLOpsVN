#!/bin/bash

# IMAGE_NAME=$1
IMAGE_TAG=$(git describe --always)


run_predictor() {
    # model_path=$2
    # path=$3
    # port=$4
    # predictor_config_path=$5
    # server=$6
    phase=$1

    echo "$phase"
    
    docker build -f deployment/drift_monitoring/Dockerfile -t drift_monitoring:$IMAGE_TAG .
        PHASE_ID=$phase IMAGE_TAG=$IMAGE_TAG\
        docker-compose -f deployment/drift_monitoring/docker-compose.yml -p drift_monitoring up -d --remove-orphans
}

run_predictor "$@"