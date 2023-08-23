import argparse
import logging
import os
import random
import time
import json
import numpy as np

import mlflow
import pandas as pd
import numpy as np
import uvicorn
import yaml
from fastapi import FastAPI, Request
from pandas.util import hash_pandas_object
from pydantic import BaseModel
from problem_config import ProblemConst, create_prob_config
from raw_data_processor import RawDataProcessor
from utils import AppConfig, AppPath
from specific_data_processing import ProcessData
from data import Data
from utils.utils import save_request_data, handle_prediction
import concurrent.futures

class Data(BaseModel):
    id: str
    rows: list
    columns: list

class Model:
    def __init__(self, config_file_path, predictor_config_path, mlflow_uri='default'):
        self.config = yaml.safe_load(open(config_file_path, "r"))
        logging.info(f"model-config: {self.config}")
        
        self.predictor_config = yaml.safe_load(open(predictor_config_path, "r"))
        logging.info(f"predictor-config: {self.predictor_config}")        

        if mlflow_uri == 'default':
            mlflow.set_tracking_uri(AppConfig.MLFLOW_TRACKING_URI)
        else:
            mlflow.set_tracking_uri(mlflow_uri)

        self.prob_config = create_prob_config(
            self.config["phase_id"], self.config["prob_id"]
        )
        self.specific_handle = self.predictor_config['specific_handle']
        # load category_index
        self.category_index = RawDataProcessor.load_category_index(self.prob_config, self.specific_handle)

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
        

        self.PREDICT_CONSTANT = self.predictor_config['PREDICT_CONSTANT']
        self.DETECT_DRIFT = self.predictor_config['DETECT_DRIFT']
        
        self.LOG_TIME = self.predictor_config['LOG_TIME']
        self.CAPTURE_DATA = self.predictor_config['CAPTURE_DATA']
        self.PROCESS_DATA = self.predictor_config['PROCESS_DATA']
        
        ### vá tạm ###
        if self.config["prob_id"] == 'prob-1':
            self.type_=0
        elif self.config["prob_id"] == 'prob-2':
            self.type_=1
            
        self.predictor_logger_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1) 

    def detect_drift(self, feature_df) -> int:
        # time.sleep(0.02)
        # return random.choice([0, 1])
        # count_dup = feature_df.groupby(feature_df.columns.to_list()).agg(count_unique = ('feature1', 'count'))
        # count_dup = count_dup[count_dup['count_unique'] > 1].shape[0]
        _, counts = np.unique(feature_df.values, return_counts=True, axis=0)
        count_dup = np.sum(counts > 1)
        res_drift = 1 if count_dup < 6 else 0
        
        return res_drift
    
    def predict_constant(self, feature_df, type_:int):
        if type_ == 0:
            prediction = np.zeros(feature_df.shape[0])
        else:
            prediction = np.full(feature_df.shape[0], fill_value=self.model.classes_[0])
            
        return prediction
    
    def log_time(self, start_time, task):
        run_time = round((time() - start_time) * 1000, 0)
        logging.info(f"{task} takes {run_time} ms")        
    
    def predict(self, data: Data):
        # logging.info(f"Running on os.getpid(): {os.getpid()}")
        
        if self.LOG_TIME:
            start_time = time()

        # preprocess
        
        raw_df = pd.DataFrame(data.rows, columns=data.columns)
        
        #======================= CAPTURE DATA =============#

        if self.CAPTURE_DATA:
            # if len(os.listdir(f"{self.prob_config.captured_data_dir}/raw/")) < 100:
                # save_request_data(
                #     raw_df, f"{self.prob_config.captured_data_dir}/raw/", data.id
                # )

            save_request_data(
                raw_df, f"{self.prob_config.captured_data_dir}/raw/", data.id
            )

        if self.specific_handle:
            raw_df = ProcessData.HANDLE_DATA[f'{self.prob_config.phase_id}_{self.prob_config.prob_id}'](raw_df, phase='test')
            cate_cols = [col for col in raw_df.columns.tolist() if raw_df[col].dtype == 'O']
        else:
            cate_cols = self.prob_config.categorical_cols
        
        if self.PROCESS_DATA:
            feature_df = RawDataProcessor.apply_category_features(
                raw_df=raw_df,
                categorical_cols=cate_cols,
                category_index=self.category_index,
            )
        else:
            feature_df = raw_df
        
        if self.LOG_TIME:
            self.predictor_logger_executor.submit(self.log_time, start_time, 'process_data')
            # start_time_ = time()
            # self.log_time(start_time, 'process_data')
            # self.log_time(start_time_, 'log')
            start_time = time()
        
        #======================= CAPTURE DATA =============#
        if self.CAPTURE_DATA:
            # if len(os.listdir(self.prob_config.captured_data_dir)) < 100:
            #     ModelPredictor.save_request_data(
            #         feature_df, self.prob_config.captured_data_dir, data.id
            #     )

            ModelPredictor.save_request_data(
                feature_df, self.prob_config.captured_data_dir, data.id
            )
            
        get_features = [each['name'] for each in self.input_schema]        
        
        if self.DETECT_DRIFT:
            res_drift = self.detect_drift(feature_df[get_features])
        else:
            res_drift = 0
    
        if self.LOG_TIME:
            self.predictor_logger_executor.submit(self.log_time, start_time, 'drift')
            # start_time_ = time()
            # self.log_time(start_time, 'drift')
            # self.log_time(start_time_, 'log')
            start_time = time()
        
        if self.PREDICT_CONSTANT:
            prediction = self.predict_constant(feature_df[get_features], type_=type_)
        else:
            if type_ == 0:
                prediction = self.model.predict_proba(feature_df[get_features])[:, 1]
            else:
                prediction = self.model.predict(feature_df[get_features])

        # logging.info(prediction)
        # res_drift = self.detect_drift(feature_df[get_features])

        if self.LOG_TIME:
            self.predictor_logger_executor.submit(self.log_time, start_time, 'prediction')
            # start_time_ = time()
            # self.log_time(start_time, 'prediction')
            # self.log_time(start_time_, 'log')
            start_time = time()
        
        # res_drift.wait()
        # res_drift = res_drift_task.get(timeout=3)
        
        return {
            "id": data.id,
            "predictions": prediction.tolist(),
            "drift": res_drift
        }

    @staticmethod
    def save_request_data(feature_df: pd.DataFrame, captured_data_dir, data_id: str):
        os.makedirs(captured_data_dir, exist_ok=True)
        if data_id.strip():
            filename = data_id
        else:
            filename = hash_pandas_object(feature_df).sum()
        output_file_path = os.path.join(captured_data_dir, f"{filename}.parquet")
        feature_df.to_parquet(output_file_path, index=False)
        return output_file_path


class PredictorApi:
    def __init__(self, predictor1: ModelPredictor, predictor2: ModelPredictor):
        self.predictor1 = predictor1
        self.predictor2 = predictor2

        self.app = FastAPI()

        @self.app.get("/")
        def root():
            return {"message": "hello"}

        @self.app.post("/phase-3/prob-1/predict")
        def predict(data: Data, request: Request):
            self._log_request(request)
            response = self.predictor1.predict(data, 0)
            self._log_response(response)
            return response
        
        @self.app.post("/phase-3/prob-2/predict")
        def predict(data: Data, request: Request):
            self._log_request(request)
            response = self.predictor2.predict(data, 1)
            self._log_response(response)
            return response


    @staticmethod
    def _log_request(request: Request):
        pass

    @staticmethod
    def _log_response(response: dict):
        pass

    def run(self, port):
        uvicorn.run("model_predictor:api.app", host="0.0.0.0", port=port, workers=16)

default_config_path = (
        AppPath.MODEL_CONFIG_DIR
        / ProblemConst.PHASE2
        / ProblemConst.PROB1
        / "model-1.yaml"
    ).as_posix()

parser = argparse.ArgumentParser()
parser.add_argument("--config-path1", type=str, default='data/model_config/phase-2/prob-1/phase-2_prob-1_cv.yaml')
parser.add_argument("--config-path2", type=str, default='data/model_config/phase-2/prob-2/phase-2_prob-2_lgbm__.yaml')
parser.add_argument("--specific_handle", type=lambda x: (str(x).lower() == "true"), default=False)
parser.add_argument("--port", type=int, default=PREDICTOR_API_PORT)
args = parser.parse_args()

predictor1 = ModelPredictor(config_file_path=args.config_path1, specific_handle=args.specific_handle)
predictor2 = ModelPredictor(config_file_path=args.config_path2, specific_handle=args.specific_handle)

api = PredictorApi(predictor1, predictor2)

if __name__ == "__main__":
    api.run(port=args.port)
