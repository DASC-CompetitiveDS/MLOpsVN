import os
from datetime import timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator, BranchPythonOperator
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

client = Minio(
        "127.0.0.1:9009",
        access_key="gGL0SPj6CNLosSNR7nfM",
        secret_key="qxViOu0AW2z6kaa7VtrOmXENOMOlQIwMoJPfTK2D",
        secure=False
    )

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
    'ci_pipeline',
    default_args=default_args,
    description='Continuous Training Pipeline',
    schedule_interval=timedelta(days=1),
)

with dag:
    pass

    minio_uri = os.environ.get("MINIO_URI")

    dowload_data = PythonOperator(task_id='download_data',
                                python_callable=folder_getter.get_data,
                                op_kwargs={'minio_server': minio_uri,
                                           'include_pattern':'phase-3'}
                                )

    # dowload_captured_data = PythonOperator(task_id='dowload_captured_data',
    #                             python_callable=storage_utils.folder_getter.get_data,
    #                             op_kwargs={'minio_server': minio_uri,
    #                                        'src_path':'data/captured_data/phase-3',
    #                                        'captured_version':'11-13',
    #                                        'drop_exist':True}
    #                             )

    prob_config = PythonOperator(
        task_id='problem_config',
        python_callable=problem_config.get_prob_config,
        op_kwargs={'phase_id': phase_id,
                   'prob_id': prob_id}
    )

    data_ingestion = PythonOperator(
        task_id='data_ingestion',
        python_callable=raw_data_processor.RawDataProcessor.process_raw_data,
        op_kwargs={'phase_id': phase_id,
                   'prob_id': prob_id,
                   'remove_dup' : 'None',
                   'order_reg': 0,
                   'drift': False,
                   'specific_handle': False,
                   'external_data': False,
                   'kfold': False}
    )
    
    baseline_model = PythonOperator(
        task_id='baseline_model',
        python_callable=model_trainer.ModelTrainer.train_model,
        op_kwargs={'phase_id': phase_id,
                   'prob_id': prob_id,
                   'task': 'clf',
                   'type_model': 'lgbm',
                   'class_weight': False,
                   'time_tuning': 20,
                   'specific_handle': False,
                   'add_captured_data': False,
                   'log_confusion_matrix': False,
                   'model_name': None,
                   'cross_validation': True,
                   'kfold': -1,
                   'learning_rate': 0.1,
                   'early_stopping_rounds': 300}
    )

    upload_data = PythonOperator(
        task_id = 'upload_data',
        python_callable = file_putter.put_data,
        op_kwargs = {'minio_server':minio_uri,
                     'path': 'data/train_data/phase-3'}
    )

    dowload_data >> prob_config >> data_ingestion >> baseline_model >> upload_data 

    