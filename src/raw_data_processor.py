import argparse
import logging
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split

from problem_config import ProblemConfig, ProblemConst, get_prob_config
from sklearn.model_selection import StratifiedKFold

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
        for col in categorical_cols:
            apply_df[col] = apply_df[col].astype("category")
            apply_df[col] = pd.Categorical(
                apply_df[col],
                categories=category_index[col],
            ).codes
        return apply_df
    
    @staticmethod
    def process_default(training_data):
        training_data, category_index = RawDataProcessor.build_category_features(
            training_data, prob_config.categorical_cols
        )
        train, dev = train_test_split(
            training_data,
            test_size=prob_config.test_size,
            random_state=prob_config.random_state,
        )

        with open(prob_config.category_index_path, "wb") as f:
            pickle.dump(category_index, f)
        target_col = prob_config.target_col 
        train_x = train.drop([target_col], axis=1)
        train_y = train[[target_col]]
        test_x = dev.drop([target_col], axis=1)
        test_y = dev[[target_col]]

        train_x.to_parquet(prob_config.train_x_path, index=False)
        train_y.to_parquet(prob_config.train_y_path, index=False)
        test_x.to_parquet(prob_config.test_x_path, index=False)
        test_y.to_parquet(prob_config.test_y_path, index=False)
    
    @staticmethod  
    def stratified_kfold(training_data, prob_config: ProblemConfig):
        X = training_data
        y = training_data[prob_config.target_col]
        skf = StratifiedKFold(n_splits=prob_config.n_folds)
        list_subsets = []
        
        for i, (train_index, test_index) in enumerate(skf.split(X, y)):
            train, dev = X.iloc[train_index], X.iloc[test_index]
            list_subsets.append([train, dev])
            
        return list_subsets
    
    @staticmethod
    def process_raw_data(prob_config: ProblemConfig, default=True):
        logging.info("start process_raw_data")
        training_data = pd.read_parquet(prob_config.raw_data_path)
        if default:
            training_data, category_index = RawDataProcessor.build_category_features(
                training_data, prob_config.categorical_cols
            )
            train, dev = train_test_split(
                training_data,
                test_size=prob_config.test_size,
                random_state=prob_config.random_state,
            )

            with open(prob_config.category_index_path, "wb") as f:
                pickle.dump(category_index, f)
            target_col = prob_config.target_col 
            train_x = train.drop([target_col], axis=1)
            train_y = train[[target_col]]
            test_x = dev.drop([target_col], axis=1)
            test_y = dev[[target_col]]

            train_x.to_parquet(prob_config.train_x_path, index=False)
            train_y.to_parquet(prob_config.train_y_path, index=False)
            test_x.to_parquet(prob_config.test_x_path, index=False)
            test_y.to_parquet(prob_config.test_y_path, index=False)
            
        else:
            pass
        
        logging.info("finish process_raw_data")

    @staticmethod
    def load_train_data(prob_config: ProblemConfig):
        train_x_path = prob_config.train_x_path
        train_y_path = prob_config.train_y_path
        train_x = pd.read_parquet(train_x_path)
        train_y = pd.read_parquet(train_y_path)
        return train_x, train_y[prob_config.target_col]

    @staticmethod
    def load_test_data(prob_config: ProblemConfig):
        dev_x_path = prob_config.test_x_path
        dev_y_path = prob_config.test_y_path
        dev_x = pd.read_parquet(dev_x_path)
        dev_y = pd.read_parquet(dev_y_path)
        return dev_x, dev_y[prob_config.target_col]

    @staticmethod
    def load_category_index(prob_config: ProblemConfig):
        with open(prob_config.category_index_path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def load_capture_data(prob_config: ProblemConfig):
        captured_x_path = prob_config.captured_x_path
        captured_y_path = prob_config.uncertain_y_path
        captured_x = pd.read_parquet(captured_x_path)
        captured_y = pd.read_parquet(captured_y_path)
        return captured_x, captured_y[prob_config.target_col]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-id", type=str, default=ProblemConst.PHASE1)
    parser.add_argument("--prob-id", type=str, default=ProblemConst.PROB1)
    args = parser.parse_args()

    prob_config = get_prob_config(args.phase_id, args.prob_id)
    RawDataProcessor.process_raw_data(prob_config)
