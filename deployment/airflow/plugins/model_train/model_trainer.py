import argparse
import logging

import os
import yaml
import mlflow
import numpy as np
from mlflow.models.signature import infer_signature
import numpy as np
import pickle
import pandas as pd

from problem_config import (
    ProblemConfig,
    ProblemConst,
    get_prob_config,
)
from data_processing.raw_data_processor import RawDataProcessor
from model_optimization import get_best_params, model_training, get_best_params_cv, AddCapturedStratifiedKFold
from utils.config import AppConfig
from utils.utils import get_confusion_matrix, get_feature_importance

import warnings
warnings.filterwarnings("ignore")



class ModelTrainer:
    @staticmethod
    def train_model(type_model, time_tuning, task, class_weight, model_name, learning_rate, early_stopping_rounds, drift_training=False, specific_handle=False, kfold=None, add_captured_data=False, cross_validation=True, log_confusion_matrix=False, **kwargs):
        logging.info("start train_model")
        ti = kwargs['ti']
        prob_config = ti.xcom_pull(task_ids='problem_config')
        # init mlflow
        if model_name is None:
            model_name = f"{prob_config.phase_id}_{prob_config.prob_id}_{type_model}"\
                        +f"{'' if class_weight is False else '_class_weight'}"\
                        +f"{'' if add_captured_data is False else '_add_captured_data'}"\
                        +f"{'' if cross_validation is False else '_cv'}"\
                        +f"{'_specific_handle' if specific_handle is True else ''}"\
                        +f"{'_drift' if drift_training is True else ''}"\
                        +f"{f'_fold{kfold}' if kfold != -1 else ''}"
        else:
            model_name = f"{prob_config.phase_id}_{prob_config.prob_id}_{type_model}_{model_name}"
                        
        mlflow.set_tracking_uri(AppConfig.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(model_name)

        # load train data
        train_x, train_y = RawDataProcessor.load_train_data(prob_config, drift_training, kfold)
        logging.info(f"loaded {len(train_x)} samples")

        # index indicates which sample is real, which is pseudo
        real_label_array = np.ones(len(train_x))
        
        if add_captured_data:
            captured_x, captured_y = RawDataProcessor.load_capture_data(prob_config)
            train_x = pd.concat([train_x, captured_x])
            train_y = pd.concat([train_y, captured_y])
            
            real_label_array = np.concatenate((real_label_array, np.zeros(len(captured_x))))
            logging.info(f"added {len(captured_x)} captured samples")
            
            
        category_features = RawDataProcessor.load_category_index(prob_config, specific_handle)
        category_features = list(category_features.keys())
            
        test_x, test_y = RawDataProcessor.load_test_data(prob_config, drift_training, kfold)        
        
        if cross_validation:
            real_label_array = np.concatenate((real_label_array, np.ones(len(test_x))))
            train_x, train_y, test_x, test_y = RawDataProcessor.combine_train_val(train_x, train_y, test_x, test_y)

        if add_captured_data:
            cv_generator = AddCapturedStratifiedKFold(n_folds=5)
            train_x, train_y = cv_generator.move_captured_to_last(train_x, train_y, origin_real_label_array=real_label_array)
        else:
            cv_generator = None
        
        # get params
        if time_tuning != 0:
            if not cross_validation:
                params_tuning = prob_config.params_tuning[type_model]
                model_params = get_best_params((train_x, train_y), (test_x, test_y), type_model, task, params_tuning, category_features, 
                                            class_weight, time_tuning, idx_phase=f"{prob_config.phase_id}_{prob_config.prob_id}", model_name=model_name)
            else:
                model_params, num_boost_round = get_best_params_cv((train_x, train_y), type_model, task, category_features, class_weight, time_tuning, learning_rate, early_stopping_rounds,
                                                  idx_phase=f"{prob_config.phase_id}_{prob_config.prob_id}", model_name=model_name, cv_generator=cv_generator)
                
        else:
            model_params = prob_config.params_fix[type_model]
        
        mlflow.start_run(run_name=model_name)
        
        # train and evaluate
        if not cross_validation:
            model, validation_score, predictions = model_training((train_x, train_y), (test_x, test_y), 
                                                                type_model, task, model_params, category_features, class_weight)
        else:
            model, validation_score, predictions = model_training((train_x, train_y), (test_x, test_y), 
                                                                type_model, task, model_params, category_features, class_weight, num_boost_round)
        key_metrics = "validation_score"
        metrics = {key_metrics: validation_score}
        logging.info(f"metrics: {metrics}")
        
        # model config yaml.file
        # mlflow log
        
        mlflow.set_tag('type_model', type_model)
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        if log_confusion_matrix:
            mlflow.log_figure(get_confusion_matrix(test_y, predictions), 
                            "confusion_matrix.png")
            
        for importance_type in ['split', 'gain']:
            fig, importance_dict = get_feature_importance(model, importance_type=importance_type)
            mlflow.log_figure(fig, f'feature_importances_{importance_type}.png')
            mlflow.log_dict(importance_dict, f"feature_importances_{importance_type}.json")
        
        signature = infer_signature(test_x, predictions)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=AppConfig.MLFLOW_MODEL_PREFIX,
            signature=signature,
            registered_model_name=model_name
        )
        mlflow.end_run()
        model_config_path = f"{prob_config.model_config_path}/{model_name}.yaml"
        model_config = {"phase_id": prob_config.phase_id, "prob_id": prob_config.prob_id, "model_name": model_name}
        client = mlflow.MlflowClient(tracking_uri=AppConfig.MLFLOW_TRACKING_URI)
        latest_vestion = int(client.get_latest_versions(model_name, stages=["None"])[0].version)
        model_config["model_version"] = latest_vestion
        with open(model_config_path, "w") as file:
            yaml.dump(model_config, file)
        logging.info(f"model name: {model_name}")
        logging.info("finish train_model")
        return model_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-id", type=str, default=ProblemConst.PHASE3)
    parser.add_argument("--prob-id", type=str, default=ProblemConst.PROB2)

    parser.add_argument("--task", type=str, default='clf', 
                        help="Tác vụ thực hiện ['clf', 'reg']")
    parser.add_argument("--type_model", type=str, default='lgbm', 
                        help='loại model sử dụng (xgb, lgbm, cb, rdf)')
    parser.add_argument("--class_weight", type=lambda x: (str(x).lower() == "true"), default=False, 
                        help='Sử dụng class weight')
    parser.add_argument("--time_tuning", type=float, default=20, 
                        help='Thời gian tuning model, nếu = 0 tức là không sử dụng')
    parser.add_argument("--drift_training", type=lambda x: (str(x).lower() == "true"), default=False, 
                        help='sử dụng dữ liệu drift')
    parser.add_argument("--specific_handle", type=lambda x: (str(x).lower() == "true"), default=False)
    parser.add_argument("--add_captured_data", type=lambda x: (str(x).lower() == "true"), default=True)
    parser.add_argument("--log_confusion_matrix", type=lambda x: (str(x).lower() == "true"), default=False)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--cross_validation", type=lambda x: (str(x).lower() == "true"), default=True)
    parser.add_argument("--kfold", type=int, default=-1)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--early_stopping_rounds", type=int, default=300)

    
    args = parser.parse_args()

    prob_config = get_prob_config(args.phase_id, args.prob_id)
    
    # mlflow.autolog()
    
    if args.type_model not in ['xgb', 'lgbm', 'cb', 'rdf']:
        print("The available model type: [xgb, lgbm, cb, rdf]")
    elif args.task not in ['clf', 'reg']:
        print("The available task: [clf, reg]")
    else:
        ModelTrainer.train_model(args,
            args.type_model, args.time_tuning, args.task, args.class_weight, model_name=args.model_name, learning_rate=args.learning_rate, early_stopping_rounds=args.early_stopping_rounds, drift_training=args.drift_training, specific_handle=args.specific_handle, kfold=args.kfold, add_captured_data=args.add_captured_data, cross_validation=args.cross_validation, log_confusion_matrix=args.log_confusion_matrix
        )