import argparse
import logging
import os
import random
import time

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

import uvloop
import asyncio
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

PREDICTOR_API_PORT = 8000


class Data(BaseModel):
    id: str
    rows: list
    columns: list


class ModelPredictor:
    def __init__(self, config_file_path):
        with open(config_file_path, "r") as f:
            self.config = yaml.safe_load(f)
        logging.info(f"model-config: {self.config}")

        mlflow.set_tracking_uri(AppConfig.MLFLOW_TRACKING_URI)

        self.prob_config = create_prob_config(
            self.config["phase_id"], self.config["prob_id"]
        )

        # load category_index
        self.category_index = RawDataProcessor.load_category_index(self.prob_config)

        # load model
        model_uri = os.path.join(
            "models:/", self.config["model_name"], str(self.config["model_version"])
        )
        self.input_schema = mlflow.models.Model.load(model_uri).get_input_schema().to_dict()
        self.model = mlflow.sklearn.load_model(model_uri)

    def detect_drift(self, feature_df) -> int:
        # time.sleep(0.02)
        # return random.choice([0, 1])
        return 1

    def predict(self, data: Data, type_: int):
        # start_time = time.time()

        # preprocess
        raw_df = pd.DataFrame(data.rows, columns=data.columns)
        feature_df = RawDataProcessor.apply_category_features(
            raw_df=raw_df,
            categorical_cols=self.prob_config.categorical_cols,
            category_index=self.category_index,
        )
        # save request data for improving models
        # ModelPredictor.save_request_data(
        #     feature_df, self.prob_config.captured_data_dir, data.id
        # )
        get_features = [each['name'] for each in self.input_schema]
        if type_ == 0:
            prediction = self.model.predict_proba(feature_df[get_features])[:, 1]
        else:
            prediction = self.model.predict(feature_df[get_features])
        # logging.info(prediction)
        # is_drifted = self.detect_drift(feature_df[get_features])

        # run_time = round((time.time() - start_time) * 1000, 0)
        # logging.info(f"prediction takes {run_time} ms")
        return {
            "id": data.id,
            "predictions": prediction.tolist(),
            "drift": 0,
        }

    @staticmethod
    def save_request_data(feature_df: pd.DataFrame, captured_data_dir, data_id: str):
        if data_id.strip():
            filename = data_id
        else:
            filename = hash_pandas_object(feature_df).sum()
        output_file_path = os.path.join(captured_data_dir, f"{filename}.parquet")
        feature_df.to_parquet(output_file_path, index=False)
        return output_file_path


class PredictorApi:
    def __init__(self, predictor1: ModelPredictor):
        self.predictor1 = predictor1

        self.app = FastAPI()

        # @self.app.get("/")
        # def root():
        #     return {"message": "hello"}

        @self.app.post("/phase-2/prob-1/predict")
        def predict(data: Data, request: Request):
            self._log_request(request)
            response = self.predictor1.predict(data, 0)
            self._log_response(response)
            return response


    @staticmethod
    def _log_request(request: Request):
        pass

    @staticmethod
    def _log_response(response: dict):
        pass

    def run(self, port):
        uvicorn.run("model_predictor_1:api.app", host="0.0.0.0", port=port, workers = 3)


default_config_path = (
        AppPath.MODEL_CONFIG_DIR
        / ProblemConst.PHASE1
        / ProblemConst.PROB1
        / "model-1.yaml"
    ).as_posix()

parser = argparse.ArgumentParser()
parser.add_argument("--config-path1", type=str, default=default_config_path)

parser.add_argument("--port", type=int, default=PREDICTOR_API_PORT)
args = parser.parse_args()

predictor1 = ModelPredictor(config_file_path=args.config_path1)

api = PredictorApi(predictor1)

if __name__ == "__main__":
    api.run(port=args.port)
