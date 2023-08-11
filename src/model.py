from time import time
import pandas as pd
import numpy as np
import uvicorn
import logging
import mlflow
import yaml
import os

from problem_config import ProblemConst, create_prob_config
from raw_data_processor import RawDataProcessor
from utils.config import AppConfig
from specific_data_processing import ProcessData
from data import Data
from utils.utils import save_request_data

LOG_TIME = False
CAPTURE_DATA = False
PROCESS_DATA = True



class Model:
    def __init__(self, config_file_path, specific_handle, PREDICT_CONSTANT=False, DETECT_DRIFT=True, mlflow_uri='default'):
        with open(config_file_path, "r") as f:
            self.config = yaml.safe_load(f)
        logging.info(f"model-config: {self.config}")

        logging.info(mlflow_uri)
        if mlflow_uri == 'default':
            mlflow.set_tracking_uri(AppConfig.MLFLOW_TRACKING_URI)
        else:
            mlflow.set_tracking_uri(mlflow_uri)

        self.prob_config = create_prob_config(
            self.config["phase_id"], self.config["prob_id"]
        )
        self.specific_handle = specific_handle
        # load category_index
        self.category_index = RawDataProcessor.load_category_index(self.prob_config, specific_handle)

        # load model
        model_uri = os.path.join(
            "models:/", self.config["model_name"], str(self.config["model_version"])
        )
        model_drift = os.path.join(
            "models:/", "phase-2_prob-2_lgbm___drift", "3"
        )
        self.input_schema = mlflow.models.Model.load(model_uri).get_input_schema().to_dict()
        self.model = mlflow.sklearn.load_model(model_uri)
        self.model_drift = mlflow.sklearn.load_model(model_drift)
        
        self.dtypes_dict = {}
        self.dtypes_dict.update({col:'f' for col in self.prob_config.feature_configs['numeric_columns']})
        self.dtypes_dict.update({col:'O' for col in self.prob_config.feature_configs['category_columns']})
        self.dtypes_dict.update({self.prob_config.feature_configs['target_column']:'O'})
        
        self.PREDICT_CONSTANT = PREDICT_CONSTANT
        self.DETECT_DRIFT = DETECT_DRIFT
        
        ### vá tạm ###
        if self.config["prob_id"] == 'prob-1':
            self.type_=0
        elif self.config["prob_id"] == 'prob-2':
            self.type_=1

    def detect_drift(self, feature_df) -> int:
        # time.sleep(0.02)
        # return random.choice([0, 1])
        count_dup = feature_df.groupby(feature_df.columns.to_list()).agg(count_unique = ('feature1', 'count'))
        count_dup = count_dup[count_dup['count_unique'] > 1].shape[0]
        res_drift = 1 if count_dup < 6 else 0
        
        return res_drift
    
    def predict_constant(self, feature_df, type_:int):
        if type_ == 0:
            prediction = np.zeros(feature_df.shape[0])
        else:
            prediction = np.full(feature_df.shape[0], fill_value=self.model.classes_[0])
            
        return prediction
    
    def predict(self, data: Data):
        # logging.info(f"Running on os.getpid(): {os.getpid()}")
        
        if LOG_TIME:
            start_time = time.time()

        # preprocess
        
        raw_df = pd.DataFrame(data.rows, columns=data.columns)
        
        #======================= CAPTURE DATA =============#

        if CAPTURE_DATA:
            if len(os.listdir(f"{self.prob_config.captured_data_dir}/raw/")) < 100:
                save_request_data(
                    raw_df, f"{self.prob_config.captured_data_dir}/raw/", data.id
                )

        if self.specific_handle:
            raw_df = ProcessData.HANDLE_DATA[f'{self.prob_config.phase_id}_{self.prob_config.prob_id}'](raw_df, phase='test')
            cate_cols = [col for col in raw_df.columns.tolist() if raw_df[col].dtype == 'O']
        else:
            cate_cols = self.prob_config.categorical_cols
        
        if PROCESS_DATA:
            feature_df = RawDataProcessor.apply_category_features(
                raw_df=raw_df,
                categorical_cols=cate_cols,
                category_index=self.category_index,
            )
        else:
            feature_df = raw_df
        
        if LOG_TIME:
            run_time = round((time.time() - start_time) * 1000, 0)
            logging.info(f"process data takes {run_time} ms")
            start_time = time.time()
        
        #======================= CAPTURE DATA =============#
        if CAPTURE_DATA:
            # if len(os.listdir(self.prob_config.captured_data_dir)) < 100:
            #     save_request_data(
            #         feature_df, self.prob_config.captured_data_dir, data.id
            #     )

            save_request_data(
                feature_df, self.prob_config.captured_data_dir, data.id
            )

            
        get_features = [each['name'] for each in self.input_schema]        
        
        if self.DETECT_DRIFT:
            res_drift = self.detect_drift(feature_df[get_features])
        else:
            res_drift = 0
    
        if LOG_TIME:
            run_time = round((time.time() - start_time) * 1000, 0)
            logging.info(f"drift takes {run_time} ms")
            start_time = time.time()
        
        if self.PREDICT_CONSTANT:
            prediction = self.predict_constant(feature_df[get_features], type_=self.type_)
        else:
            if self.type_ == 0:
                prediction = self.model.predict_proba(feature_df[get_features])[:, 1]
            else:
                prediction = self.model.predict(feature_df[get_features])
        # logging.info(prediction)
        # res_drift = self.detect_drift(feature_df[get_features])

        if LOG_TIME:
            run_time = round((time.time() - start_time) * 1000, 0)
            logging.info(f"prediction takes {run_time} ms")
            start_time = time.time()
        
        # res_drift.wait()
        
        # res_drift = res_drift_task.get(timeout=3)
        
        return {
            "id": data.id,
            "predictions": prediction.tolist(),
            "drift": res_drift
        }
