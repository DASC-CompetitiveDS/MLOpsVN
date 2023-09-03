import os
from datetime import timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator, BranchPythonOperator
from airflow.operators.docker_operator import DockerOperator
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.dates import days_ago
from airflow.utils.timezone import datetime
from data_upload import upload_files
from data_processing import label_captured_data
from data_processing import raw_data_processor
from model_train import model_trainer
from storage_utils import folder_getter, file_putter
from minio import Minio
from minio.error import S3Error
import problem_config

phase_id = 'phase-3'
prob_id = 'prob-1'

raw_data_dir = '/data/raw_data'
train_dir = f'/data/train_data/{phase_id}/{prob_id}'

root_dir = "/"
data_dir = "/{}"


if not root_dir:
    raise ValueError('PROJECT_PATH environment variable not set')

default_args = {
    'owner': 'v_users',
    'depends_on_past': False,
    'start_date': days_ago(0),
    'retries': 1,
    'retry_delay': timedelta(seconds=5)
}

dag = DAG(
    'cd_pipeline',
    default_args=default_args,
    description='Continuous Deployment Pipeline',
    schedule_interval=timedelta(days=1),
)

with dag:
    pass

    minio_uri = os.environ.get("MINIO_URI")

    get_config = PythonOperator(
        task_id = 'get_config_dag',
        python_callable=folder_getter.get_data,
                                op_kwargs={'minio_server': minio_uri,
                                           'src_path':'data/predictor_config'}                        
    )

    # kill_old_predictor = BashOperator(
    #     task_id='kill_old_predictor_dag',
    #     bash_command="docker kill model1-model_predictor-1 | docker kill model2-model_predictor-1",
    # )

    build_new_predictor = DockerOperator(
        task_id='build_new_predictor_dag',
        # bash_command="$PROJECT_PATH/model_predictor/deploy.sh model1 data/model_config/phase-3/prob-1/{{ ti.xcom_pull(task_ids='baseline_model') }} /phase-3/prob-1/predict 5001 data/predictor_config/default_log.yaml default",
        # do_xcom_push=True
        image='model1',
        container_name='model1-model_predictor-1',
        api_version='auto',
        auto_remove=True,
        entrypoint="/bin/sh -c \"python3 utils/show_parquet.py\"",
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge"
    )

    get_config >> build_new_predictor

    