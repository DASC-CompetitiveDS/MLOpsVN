import argparse
import logging
import os
import random
import time
import json
import numpy as np

import mlflow
import pandas as pd
import uvicorn
import yaml
from fastapi import FastAPI, Request
from pandas.util import hash_pandas_object

import threading
import uvloop
import asyncio
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

import model
PREDICTOR_API_PORT = 8000

class PredictorApi:
    def __init__(self, predictor: ModelPredictor):
        self.predictor1 = predictor1
        self.predictor2 = predictor2

        self.app = FastAPI()


        @self.app.post("/phase-2/prob-2/predict")
        def predict(data: Data, request: Request):
            self._log_request(request)
            response = self.predictor1.predict(data, 0)
            self._log_response(response)
            return response
        
        # @self.app.post("/phase-2/prob-2/predict")
        # def predict(data: Data, request: Request):
        #     self._log_request(request)
        #     response = self.predictor2.predict(data, 1)
        #     self._log_response(response)
            # return response


    @staticmethod
    def _log_request(request: Request):
        pass

    @staticmethod
    def _log_response(response: dict):
        pass

    def run(self, port):
        uvicorn.run("model_predictor:api.app", host="0.0.0.0", port=port, workers=8)

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
