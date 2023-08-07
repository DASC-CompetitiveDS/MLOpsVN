import uvicorn
import yaml
import os

from fastapi import FastAPI, Request
from pandas.util import hash_pandas_object

from model import Model
from data import Data


class PredictorApi:
    def __init__(self, predictor: Model, path: str):
        self.predictor = predictor

        self.app = FastAPI()

        @self.app.post(path)
        def predict(data: Data, request: Request):
            self._log_request(request)
            response = self.predictor.predict(data)
            self._log_response(response)
            return response


    @staticmethod
    def _log_request(request: Request):
        pass

    @staticmethod
    def _log_response(response: dict):
        pass

    def run(self, file, port):
        uvicorn.run(f"{file}:predictor.app", host="0.0.0.0", port=port, workers=8)