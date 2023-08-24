import argparse
import os
import uvloop
import asyncio
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

from model import Model
from data import Data
from predictor_api import PredictorApi


parser = argparse.ArgumentParser()
parser.add_argument("--config-path", type=str, default='data/model_config/phase-3/prob-1/phase-3_prob-1_cv.yaml')
parser.add_argument("--path", type=str, default='/phase-3/prob-1/predict')
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--mlflow_uri", type=str, default='default')
parser.add_argument("--predictor-config-path", type=str, default='data/predict_config/default.yaml')

args = parser.parse_args()

model = Model(config_file_path=args.config_path, mlflow_uri=args.mlflow_uri, predictor_config_path=args.predictor_config_path)
predictor = PredictorApi(model, args.path)


if __name__ == "__main__":
    file = __file__.split("/")[-1].split(".")[0]
    predictor.run(file=file, port=args.port)