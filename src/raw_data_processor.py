import argparse
import logging
import pickle
import json
import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from problem_config import ProblemConfig, ProblemConst, get_prob_config
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder  
from specific_data_processing import ProcessData

from utils.config import AppConfig
import mlflow
import yaml
import warnings
warnings.filterwarnings('ignore')

class TargetEncoder:
    def __init__(self) -> None:
        self._values = {}
    def fit(self, data, categorical_col, target):
        self._values = data[[categorical_col, target]].groupby(categorical_col)[target].mean().to_dict()
    def transform(self, series):
        return series.map(self._values)
    def fit_transform(self, data, categorical_col, target):
        self.fit(data, categorical_col, target)
        return self.transform(data[categorical_col])

class RawDataProcessor:
    @staticmethod
    def build_category_features(data, categorical_cols=None):
        if categorical_cols is None:
            categorical_cols = []
        category_index = {}
        if len(categorical_cols) == 0:
            return data, category_index

        df = data.copy()
        # process category features
        for col in categorical_cols:
            df[col] = df[col].astype("category")
            category_index[col] = df[col].cat.categories
            df[col] = df[col].cat.codes
        return df, category_index

    @staticmethod
    def apply_category_features(
        raw_df, categorical_cols=None, category_index: dict = None
    ):
        if categorical_cols is None:
            categorical_cols = []
        if len(categorical_cols) == 0:
            return raw_df

        apply_df = raw_df.copy()
        apply_df
        for col in categorical_cols:
            apply_df[col] = apply_df[col].astype("category")
            apply_df[col] = pd.Categorical(
                apply_df[col],
                categories=category_index[col],
            ).codes
        # apply_df[categorical_cols] = apply_df[categorical_cols].astype('category')

        # # apply_cat_func = lambda col: pd.Categorical(col, categories=category_index[col]).codes    
        # def apply_cat_func(col):
        #     # print(type(col))
        #     print(col)
        #     return pd.Categorical(col, categories=category_index[col]).codes
        
        # apply_df[categorical_cols] = apply_df[categorical_cols].transform(apply_cat_func)
        
        return apply_df
    
    @staticmethod
    def remove_dup_absolutely_records(X_train, target_col, order_req):
        feature_group = X_train.columns.tolist()
        dup_ = X_train.groupby(feature_group).agg(count_per_label=(target_col, "count")).reset_index()
        feature_group.remove('label')
        count_record = X_train.groupby(feature_group).agg(count_distinct_label=(target_col, "nunique")).reset_index()
        dup_['order'] = dup_.sort_values(['count_per_label', target_col], ascending=[False, False]).groupby(feature_group).cumcount()
        dup_['count_per_label_lag'] = dup_.sort_values(['order']).groupby(feature_group)['count_per_label'].shift(-1)
        dup_ = dup_.merge(count_record, how='inner', on=feature_group)
        total_dup = dup_[dup_['count_distinct_label'] == 1].reset_index(drop=True)
        if order_req != -1:
            dup_overlap_label = dup_[(dup_['count_distinct_label'] != 1) & (dup_['order'] == order_req) & ((dup_["count_per_label"]) != (dup_["count_per_label_lag"]))].reset_index(drop=True)
            total_dup = pd.concat([total_dup, dup_overlap_label]).reset_index(drop=True)
        # total_x_merge = X_train.merge(total_dup, on=X_train.columns.tolist(), how='inner')
        return total_dup[X_train.columns.tolist()]

    @staticmethod
    def remove_dup_relatively_records(X_train, target_col):
        feature_group = X_train.columns.tolist()
        feature_group.remove(target_col)
        dup_ = X_train.groupby(feature_group).agg(nunique_label=(target_col, "nunique")).reset_index()
        X_train_final = X_train.merge(dup_, how = "inner", on=feature_group)
        X_train_final = X_train_final[X_train_final["nunique_label"] == 1].reset_index(drop=True)
        X_train_final = X_train_final.drop(columns=["nunique_label"])
        return X_train_final
    
    @staticmethod
    def split_train_validation(training_data, prob_config, target_col, drift):
        train_x, test_x, train_y, test_y = train_test_split(
            training_data.drop(columns=[target_col]),
            training_data[target_col],
            test_size=prob_config.test_size,
            random_state=prob_config.random_state,
            stratify=training_data[target_col]
        )
        train_x = train_x.reset_index(drop=True)
        test_x = test_x.reset_index(drop=True)
        train_y = train_y.reset_index(drop=False).drop(columns=['index'])
        test_y = test_y.reset_index(drop=False).drop(columns=['index'])

        train_x.to_parquet(prob_config.train_x_path if drift is False else prob_config.train_x_drift_path, index=False)
        train_y.to_parquet(prob_config.train_y_path if drift is False else prob_config.train_y_drift_path, index=False)
        test_x.to_parquet(prob_config.test_x_path if drift is False else prob_config.test_x_drift_path, index=False)
        test_y.to_parquet(prob_config.test_y_path if drift is False else prob_config.test_y_drift_path, index=False)
    
        
    @staticmethod  
    def split_stratified_kfold(training_data, prob_config, target_col):
        X = training_data.drop(columns=[target_col])
        y = training_data[target_col].values

        skf = StratifiedKFold(n_splits=5, random_state=prob_config.random_state, shuffle=True)   

        for i, (train_index, test_index) in enumerate(skf.split(X, y)):
            X_train = X.iloc[train_index].reset_index(drop=True)
            X_valid = X.iloc[test_index].reset_index(drop=True)

            y_train = pd.DataFrame(y[train_index], columns=[target_col])
            y_valid = pd.DataFrame(y[test_index], columns=[target_col])

            X_train.to_parquet(f"{str(prob_config.train_data_path)}/train_x_{i}.parquet", index=False)
            y_train.to_parquet(f"{str(prob_config.train_data_path)}/train_y_{i}.parquet", index=False)
            X_valid.to_parquet(f"{str(prob_config.train_data_path)}/test_x_{i}.parquet", index=False)
            y_valid.to_parquet(f"{str(prob_config.train_data_path)}/test_y_{i}.parquet", index=False)
    
    
    @staticmethod
    def get_model_predictions(prob_config, training_data):
        mlflow.set_tracking_uri(AppConfig.MLFLOW_TRACKING_URI)
        with open(prob_config.add_features_model, "r") as f:
                config_model = yaml.safe_load(f)
        model_uri = os.path.join(
            "models:/", config_model["model_name"], str(config_model["model_version"])
        )
        input_schema = mlflow.models.Model.load(model_uri).get_input_schema().to_dict()
        model = mlflow.sklearn.load_model(model_uri)
        return model.predict_proba(training_data[[each['name'] for each in input_schema]])[:, 1]

    @staticmethod
    def process_raw_data(prob_config: ProblemConfig, remove_dup: str, order_reg: bool, specific_handle: bool, drift: bool, external_data: bool, kfold: bool, add_features: bool):
        logging.info(f"start process_raw_data{' - drift data' if drift is True else ''}")
        training_data = pd.read_parquet(prob_config.raw_data_path)
        target_col = prob_config.target_col  
        lbc = LabelEncoder().fit(training_data[target_col])
        dict_convert = {}
        dict_convert['l2i'] = {each: idx for idx, each in enumerate(lbc.classes_)}
        dict_convert['i2l'] = {idx: each for idx, each in enumerate(lbc.classes_)}
        with open(prob_config.dict_convert_path, 'wb') as file_:
            pickle.dump(dict_convert, file_)
        if external_data:
            external_data_train = pd.read_parquet(prob_config.external_data_path)
            training_data = pd.concat([training_data, external_data_train[training_data.columns.tolist()]]).reset_index(drop=True)
        if drift is False:
            if remove_dup == 'rel':
                training_data = RawDataProcessor.remove_dup_relatively_records(training_data.copy(), target_col)
                logging.info(training_data.shape)
            elif remove_dup == 'abs':
                training_data = RawDataProcessor.remove_dup_absolutely_records(training_data.copy(), target_col, order_reg)
                logging.info(training_data.shape)
        else:
            training_data = RawDataProcessor.remove_dup_absolutely_records(training_data.copy(), target_col, 1)
        
        if specific_handle is True:
            training_data = ProcessData.HANDLE_DATA[f'{prob_config.phase_id}_{prob_config.prob_id}'](training_data, target_col)

        if specific_handle is False:
            training_data, category_index = RawDataProcessor.build_category_features(
                training_data, prob_config.categorical_cols
            )
        else:
            training_data, category_index = RawDataProcessor.build_category_features(
                training_data, [col for col in training_data.columns.tolist() if training_data[col].dtype == 'O' and target_col != col]
            )
        category_index_path = prob_config.category_index_path if specific_handle is False else prob_config.category_index_path_specific_handling
        with open(category_index_path, "wb") as f:
            pickle.dump(category_index, f)
        
        if add_features:
            training_data['feature_pred'] = RawDataProcessor.get_model_predictions(prob_config, training_data.copy())

        if kfold is False:
            RawDataProcessor.split_train_validation(training_data, prob_config, target_col, drift)
        else:
            RawDataProcessor.split_stratified_kfold(training_data, prob_config, target_col)
       
        logging.info("finished process_raw_data")

    @staticmethod
    def load_train_data(prob_config: ProblemConfig, drift: bool, kfold: int):
        if kfold == -1:
            train_x_path = prob_config.train_x_path if drift is False else prob_config.train_x_drift_path
            train_y_path = prob_config.train_y_path if drift is False else prob_config.train_y_drift_path
        else:
            train_x_path = f"{str(prob_config.train_data_path)}/train_x_{kfold}.parquet"
            train_y_path = f"{str(prob_config.train_data_path)}/train_y_{kfold}.parquet"
        train_x = pd.read_parquet(train_x_path)
        train_y = pd.read_parquet(train_y_path)
        return train_x, train_y[prob_config.target_col]

    @staticmethod
    def load_test_data(prob_config: ProblemConfig, drift: bool, kfold: int):
        if kfold == -1:
            dev_x_path = prob_config.test_x_path if drift is False else prob_config.test_x_drift_path
            dev_y_path = prob_config.test_y_path if drift is False else prob_config.test_y_drift_path
        else:
            dev_x_path = f"{str(prob_config.train_data_path)}/test_x_{kfold}.parquet"
            dev_y_path = f"{str(prob_config.train_data_path)}/test_y_{kfold}.parquet"
        dev_x = pd.read_parquet(dev_x_path)
        dev_y = pd.read_parquet(dev_y_path)
        return dev_x, dev_y[prob_config.target_col]

    @staticmethod
    def load_category_index(prob_config: ProblemConfig, specific_handle=False):
        category_index_path = prob_config.category_index_path if specific_handle is False else prob_config.category_index_path_specific_handling
        with open(category_index_path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def load_capture_data(prob_config: ProblemConfig):
        captured_x_path = prob_config.captured_x_path
        captured_y_path = prob_config.uncertain_y_path
        captured_x = pd.read_parquet(captured_x_path)
        captured_y = pd.read_parquet(captured_y_path)
        return captured_x, captured_y[prob_config.target_col]

    @staticmethod
    def combine_train_val(train_x, train_y, test_x, test_y):
        train_x = pd.concat([train_x, test_x])
        train_y = pd.concat([train_y, test_y])    
        
        test_x = train_x
        test_y = train_y
        
        return train_x, train_y, test_x, test_y    
    
    @staticmethod
    def load_dict_predict(prob_config: ProblemConfig, list_model, feature_columns):
        link_data_test = prob_config.captured_data_dir
        captured_x = pd.DataFrame()
        for file_path in os.listdir(link_data_test):
            if 'parquet' not in file_path:
                continue
            captured_data = pd.read_parquet(f'{link_data_test}/{file_path}')
            count_dup = captured_data[feature_columns].groupby(feature_columns).agg(count_unique = ('feature1', 'count'))
            count_dup = count_dup[count_dup['count_unique'] > 1].shape[0]
            captured_x = pd.concat([captured_x, captured_data])
        captured_x = captured_x[feature_columns].reset_index(drop=True)
        cup = captured_x.groupby(feature_columns).agg(count_unique = ('feature1', 'count')).reset_index().drop(columns=['count_unique'])
        list_pred_proba = []
        for model in list_model:
            pred = model.predict_proba(cup.values)
            list_pred_proba.append(pred)
        list_pred_proba = np.mean(list_pred_proba, axis=0)
        class_ = list_model[0].classes_
        list_pred_proba = class_[np.argmax(list_pred_proba, axis=1)]
        dict_predict = {}
        save_cup = cup.values
        for idx in range(len(save_cup)):
            dict_predict[tuple(save_cup[idx])] = list_pred_proba[idx]
        return dict_predict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-id", type=str, default=ProblemConst.PHASE1)
    parser.add_argument("--prob-id", type=str, default=ProblemConst.PROB1)
    parser.add_argument("--remove_dup", type=str, default="None", 
                        help="Loại bỏ bản ghi duplicate nhưng có nhiều nhãn")
    parser.add_argument("--order_reg", type=int, default=0)
    parser.add_argument("--drift", type=lambda x: (str(x).lower() == "true"), default=False, 
                         help='Tạo dữ liệu drift')
    parser.add_argument("--specific_handle", type=lambda x: (str(x).lower() == "true"), default=False)
    parser.add_argument("--external_data", type=lambda x: (str(x).lower() == "true"), default=False)
    parser.add_argument("--kfold", type=lambda x: (str(x).lower() == "true"), default=False)
    parser.add_argument("--add_features", type=lambda x: (str(x).lower() == "true"), default=False)
    args = parser.parse_args()

    prob_config = get_prob_config(args.phase_id, args.prob_id)
    if args.remove_dup not in ['abs', 'rel', 'None']:
        print("The available removing duplicate records methods: [abs, rel, None]")
    else:
        RawDataProcessor.process_raw_data(prob_config, args.remove_dup, args.order_reg, args.specific_handle, args.drift, args.external_data, args.kfold, args.add_features)
