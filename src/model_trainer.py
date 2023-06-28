import argparse
import logging

import os
import yaml
import mlflow
import numpy as np
import xgboost as xgb
from mlflow.models.signature import infer_signature
from sklearn.metrics import roc_auc_score
import numpy as np
import pickle

from problem_config import (
    ProblemConfig,
    ProblemConst,
    get_prob_config,
)
from raw_data_processor import RawDataProcessor
from model_optimization import get_best_params, model_training
from utils import AppConfig

import warnings
warnings.filterwarnings("ignore")

mlflow.autolog()

class ModelTrainer:
    @staticmethod
    def train_model(prob_config: ProblemConfig, type_model, time_tuning, task, class_weight, add_captured_data=False):
        logging.info("start train_model")
        # init mlflow
        model_name = f"{prob_config.phase_id}_{prob_config.prob_id}_{type_model}_{'' if class_weight is False else 'class_weight'}"
        mlflow.set_tracking_uri(AppConfig.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(model_name)

        # load train data
        train_x, train_y = RawDataProcessor.load_train_data(prob_config)
        # train_x = train_x.to_numpy()
        # train_y = train_y.to_numpy()
        logging.info(f"loaded {len(train_x)} samples")

        if add_captured_data:
            captured_x, captured_y = RawDataProcessor.load_capture_data(prob_config)
            captured_x = captured_x.to_numpy()
            captured_y = captured_y.to_numpy()
            train_x = np.concatenate((train_x, captured_x))
            train_y = np.concatenate((train_y, captured_y))
            logging.info(f"added {len(captured_x)} captured samples")
        
        with open(prob_config.category_index_path, "rb") as file:
            category_features = pickle.load(file)
        category_features = list(category_features.keys())
            
        test_x, test_y = RawDataProcessor.load_test_data(prob_config)  
        
        #get params
        if time_tuning != 0:
            params_tuning = prob_config.params_tuning[type_model]
            model_params = get_best_params((train_x, train_y), (test_x, test_y), type_model, task, params_tuning, category_features, 
                                           class_weight, time_tuning, idx_phase=f"{prob_config.phase_id}_{prob_config.prob_id}")
        else:
            model_params = prob_config.params_fix[type_model]
            
        
        # train and evaluate
        model, validation_score, predictions = model_training((train_x, train_y), (test_x, test_y), 
                                                              type_model, task, model_params, category_features, class_weight)
        key_metrics = "validation_auc" if task == 'clf' else "validation_rmse"
        metrics = {key_metrics: validation_score}
        logging.info(f"metrics: {metrics}")
        
        #model config yaml.file
        model_config_path = f"{prob_config.model_config_path}/{model_name}.yaml"
        if os.path.exists(model_config_path):
            with open(model_config_path, "r") as f:
                model_config = yaml.safe_load(f)
            model_config["model_version"] += 1
        else:
            model_config = {"phase_id": prob_config.phase_id, "prob_id": prob_config.prob_id, "model_name": model_name, "model_version": 1}
        with open(model_config_path, "w") as file:
            yaml.dump(model_config, file)
            
        # mlflow log
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        signature = infer_signature(test_x, predictions)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=AppConfig.MLFLOW_MODEL_PREFIX,
            signature=signature,
            registered_model_name=model_name
        )
        mlflow.end_run()
        logging.info(f"model name: {model_name}")
        logging.info("finish train_model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-id", type=str, default=ProblemConst.PHASE1)
    parser.add_argument("--prob-id", type=str, default=ProblemConst.PROB1)

    parser.add_argument("--task", type=str, default='clf', 
                        help="Tác vụ thực hiện ['clf', 'reg']")
    parser.add_argument("--type_model", type=str, default='lgbm', 
                        help='loại model sử dụng (xgb, lgbm, cb, rdf)')
    parser.add_argument("--class_weight", type=lambda x: (str(x).lower() == "true"), default=False, 
                        help='Sử dụng class weight')
    parser.add_argument("--time_tuning", type=float, default=0, 
                        help='Thời gian tuning model, nếu = 0 tức là không sử dụng')
    parser.add_argument("--add-captured-data", type=lambda x: (str(x).lower() == "true"), default=False)
    args = parser.parse_args()

    prob_config = get_prob_config(args.phase_id, args.prob_id)
    if args.type_model not in ['xgb', 'lgbm', 'cb', 'rdf']:
        print("The available model type: [xgb, lgbm, cb, rdf]")
    elif args.task not in ['clf', 'reg']:
        print("The available task: [clf, reg]")
    else:
        ModelTrainer.train_model(
            prob_config, args.type_model, args.time_tuning, args.task, args.class_weight, add_captured_data=args.add_captured_data
        )
