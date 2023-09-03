airflow_up() {
    AIRFLOW_PROJ_DIR=/home/$USER/airflow \
    docker-compose -f deployment/airflow/docker-compose.yml up
}


airflow_up "$@"