from pydantic import BaseModel
from time import time
import pandas as pd
import logging
import mlflow
import yaml
import os

from problem_config import ProblemConst, create_prob_config
from raw_data_processor import RawDataProcessor
from utils.config import AppConfig
from specific_data_processing import ProcessData

LOG_TIME = False

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

        # if self.prob_config.prob_id == "prob-2":
        #     list_model = []
        #     for i in range(5):
        #         model_uri = os.path.join(
        #             "models:/", f"phase-2_prob-2_lgbm_fold{i}", "3"
        #         )
        #         input_schema = mlflow.models.Model.load(model_uri).get_input_schema().to_dict()
        #         model = mlflow.sklearn.load_model(model_uri)
        #         list_model.append(model)
            
        #     self.dict_predict = RawDataProcessor.load_dict_predict(self.prob_config, list_model, [each['name'] for each in self.input_schema])
        # else:
        #     self.dict_predict = None
        # logging.info(self.dict_predict)

    def detect_drift(self, feature_df) -> int:
        # time.sleep(0.02)
        # return random.choice([0, 1])
        count_dup = feature_df.groupby(feature_df.columns.to_list()).agg(count_unique = ('feature1', 'count'))
        count_dup = count_dup[count_dup['count_unique'] > 1].shape[0]
        res_drift = 1 if count_dup > 400 else 0
        
        return res_drift
    
    def predict(self, data: Data, type_: int):
        # logging.info(f"Running on os.getpid(): {os.getpid()}")
        
        if LOG_TIME:
            start_time = time.time()

        # preprocess
        raw_df = pd.DataFrame(data.rows, columns=data.columns)

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
        
        if LOG_TIME:
            run_time = round((time.time() - start_time) * 1000, 0)
            logging.info(f"process data takes {run_time} ms")
            start_time = time.time()
        
        #======================= CAPTURE DATA =============#
        # if len(os.listdir(self.prob_config.captured_data_dir)) < 100:
        #     ModelPredictor.save_request_data(
        #         feature_df, self.prob_config.captured_data_dir, data.id
        #     )
            
        get_features = [each['name'] for each in self.input_schema]
        
        # print(get_features)
        # print(feature_df)
        
        # count_dup = feature_df[get_features].groupby(get_features).agg(count_unique = ('feature1', 'count'))
        # count_dup = count_dup[count_dup['count_unique'] > 1].shape[0]
        # res_drift = 1 if count_dup > 200 else 0
        
        # pool = Pool(processes=1)              # Start a worker processes.
        # res_drift_task = pool.apply_async(self.detect_drift, [feature_df[get_features]])
        feature_df = feature_df[get_features]
        res_drift = self.detect_drift(feature_df)
        
        if LOG_TIME:
            run_time = round((time.time() - start_time) * 1000, 0)
            logging.info(f"drift takes {run_time} ms")
            start_time = time.time()
        
        if type_ == 0:
            # prediction = prediction[:, 1]
            prediction = self.model.predict_proba(feature_df)[:, 1]
        else:
            # class_ = self.model.classes_
            # prediction = class_[np.argmax(prediction, axis=1)]
            prediction = self.model.predict(feature_df)
            
            # prediction = []
            # feature_df = feature_df.applymap(lambda x: float(int(x)) if round(x - int(x), 5) == 0 and abs(x) >= 1 else round(x, 6))
            # save_np = feature_df.values
            # for idx in range(len(save_np)):
            #     prediction.append(self.dict_predict[tuple(save_np[idx])])
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
