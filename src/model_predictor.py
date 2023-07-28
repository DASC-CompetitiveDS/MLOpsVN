import argparse
import logging
import os
import random
import time
import json

import mlflow
import pandas as pd
import uvicorn
import yaml
from fastapi import FastAPI, Request
from pandas.util import hash_pandas_object
from pydantic import BaseModel
from problem_config import ProblemConst, create_prob_config
from raw_data_processor import RawDataProcessor
from utils import AppConfig, AppPath
from specific_data_processing import ProcessData

import uvloop
import asyncio
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

PREDICTOR_API_PORT = 8000


class Data(BaseModel):
    id: str
    rows: list
    columns: list


class ModelPredictor:
    def __init__(self, config_file_path, specific_handle):
        with open(config_file_path, "r") as f:
            self.config = yaml.safe_load(f)
        logging.info(f"model-config: {self.config}")

        mlflow.set_tracking_uri(AppConfig.MLFLOW_TRACKING_URI)

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

    def detect_drift(self, feature_df) -> int:
        # time.sleep(0.02)
        # return random.choice([0, 1])
        return 1

    def predict(self, data: Data, type_: int):
        # start_time = time.time()

        # preprocess
        raw_df = pd.DataFrame(data.rows, columns=data.columns)

        #======================= CAPTURE DATA =============#

        if len(os.listdir(f"{self.prob_config.captured_data_dir}/raw/")) < 100:
            ModelPredictor.save_request_data(
                raw_df, f"{self.prob_config.captured_data_dir}/raw/", data.id
            )

        if self.specific_handle:
            raw_df = ProcessData.HANDLE_DATA[[f'{self.prob_config.phase_id}_{self.prob_config.prob_id}']](raw_df, phase='test')
            cate_cols = [col for col in raw_df.columns.tolist() if raw_df[col].dtype == 'O']
        else:
            cate_cols = self.prob_config.categorical_cols
        
        feature_df = RawDataProcessor.apply_category_features(
            raw_df=raw_df,
            categorical_cols=cate_cols,
            category_index=self.category_index,
        )
        
        #======================= CAPTURE DATA =============#
        if len(os.listdir(self.prob_config.captured_data_dir)) < 100:
            ModelPredictor.save_request_data(
                feature_df, self.prob_config.captured_data_dir, data.id
            )
            
        get_features = [each['name'] for each in self.input_schema]
        
        count_dup = feature_df[get_features].groupby(get_features).agg(count_unique = ('feature1', 'count'))
        count_dup = count_dup[count_dup['count_unique'] > 1].shape[0]
        res_drift = 1 if count_dup > 200 else 0
        
        if type_ == 0:
            prediction = self.model.predict_proba(feature_df[get_features])[:, 1]
        else:
            prediction = self.model.predict(feature_df[get_features])
        # logging.info(prediction)
        # res_drift = self.detect_drift(feature_df[get_features])

        # run_time = round((time.time() - start_time) * 1000, 0)
        # logging.info(f"prediction takes {run_time} ms")
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

        @self.app.post("/phase-2/prob-1/predict")
        def predict(data: Data, request: Request):
            self._log_request(request)
            response = self.predictor1.predict(data, 0)
            self._log_response(response)
            return response
        
        @self.app.post("/phase-2/prob-2/predict")
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
        uvicorn.run("model_predictor:api.app", host="0.0.0.0", port=port, workers = 6)

default_config_path = (
        AppPath.MODEL_CONFIG_DIR
        / ProblemConst.PHASE1
        / ProblemConst.PROB1
        / "model-1.yaml"
    ).as_posix()

parser = argparse.ArgumentParser()
parser.add_argument("--config-path1", type=str, default=default_config_path)
parser.add_argument("--config-path2", type=str, default=default_config_path)
parser.add_argument("--specific_handle", type=lambda x: (str(x).lower() == "true"), default=False)
parser.add_argument("--port", type=int, default=PREDICTOR_API_PORT)
args = parser.parse_args()

predictor1 = ModelPredictor(config_file_path=args.config_path1, specific_handle=args.specific_handle)
predictor2 = ModelPredictor(config_file_path=args.config_path2, specific_handle=args.specific_handle)

api = PredictorApi(predictor1, predictor2)

if __name__ == "__main__":
    api.run(port=args.port)
