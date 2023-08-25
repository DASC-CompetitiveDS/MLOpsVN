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
from utils.utils import save_request_data, handle_prediction
import concurrent.futures
from storage_utils.folder_getter import get_data


class Model:
    def __init__(self, config_file_path, predictor_config_path, server='local'):
        self.config = yaml.safe_load(open(config_file_path, "r"))
        logging.info(f"model-config: {self.config}")
        
        self.predictor_config = yaml.safe_load(open(predictor_config_path, "r"))
        logging.info(f"predictor-config: {self.predictor_config}")        

        if server=='local':
            mlflow.set_tracking_uri(AppConfig.MLFLOW_TRACKING_URI)
        else:
            mlflow.set_tracking_uri(os.path.join(server, 'mlflow/'))
            # get_data(minio_server=server.replace('http://', ''), dst_path='data')
            
        self.prob_config = create_prob_config(
            self.config["phase_id"], self.config["prob_id"]
        )
        
        self.specific_handle = self.predictor_config['specific_handle']
        # load category_index
        self.category_index = RawDataProcessor.load_category_index(self.prob_config, True if self.config['prob_id'] == 'prob-3' else False)
        # self.category_index = RawDataProcessor.load_category_index(self.prob_config, specific_handle)

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
        # res_drift = 1 if count_dup == 2 and feature_df['feature2'].loc[0] == 118 and feature_df['feature4'].loc[1] == 1 else 0
        check_thres = (feature_df['feature4'].value_counts() / feature_df.shape[0]).to_dict()[4]
        res_drift = 1 if check_thres < 0.2 else 0
        # logging.info(res_drift) 
        return res_drift
    
    def predict_constant(self, feature_df, type_:int):
        if type_ == 0:
            prediction = np.zeros(feature_df.shape[0])
        else:
            prediction = np.full(feature_df.shape[0], fill_value=self.model.classes_[5])
            
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

        if self.prob_config.prob_id == 'prob-3':
            raw_df = ProcessData.HANDLE_DATA[f'{self.prob_config.phase_id}_{self.prob_config.prob_id}'](raw_df, self.prob_config.target_col, phase='test')
            cate_cols = [col for col in raw_df.columns.tolist() if raw_df[col].dtype == 'O']
            logging.info(cate_cols)
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
            #     save_request_data(
            #         feature_df, self.prob_config.captured_data_dir, data.id
            #     )

            save_request_data(
                feature_df, self.prob_config.captured_data_dir, data.id
            )
            
        get_features = [each['name'] for each in self.input_schema]        
        
        if self.DETECT_DRIFT:
            try:
                get_features.remove('feature_pred')
            except:
                pass
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
            prediction = self.predict_constant(feature_df[get_features], type_=self.type_)
        else:
            if self.type_ == 0:
                prediction = self.model.predict_proba(feature_df[get_features])[:, 1]
                # res_ = []
                # for each in prediction:
                #     app = each
                #     if each >= 0.95:
                #         app = 1
                #     res_.append(app)
                # prediction = np.array(res_)
            else:
                # pred = feature_df[get_features]
                # pred['feature_pred'] = RawDataProcessor.get_model_predictions(self.prob_config, pred)
                # prediction = self.model.predict(pred)
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
