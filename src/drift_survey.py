import pandas as pd
import numpy as np
from raw_data_processor import RawDataProcessor
from problem_config import ProblemConfig, ProblemConst, get_prob_config
import os
from utils.config import AppConfig

def test_drift_numeric():
    pass

def test_drift_categorical(dict_org_values):
    pass

def process_dict_values(dict_, test_in_keys=None):
    new_dict_, total_other_keys = {}, 0
    for key, value in dict_.items():
        if (value > 0.01 and test_in_keys is None) or (test_in_keys is not None and key in test_in_keys):
            new_dict_[key] = value 
        else:
            total_other_keys += value
    new_dict_['other'] = total_other_keys
    return new_dict_

def compare_dict(dict_true, dict_parquet):
    total_diff = 0
    for key in dict_true:
        total_diff += abs(dict_true[key] - dict_parquet[key])
    return total_diff

def drift_survey_processing(phase_id, prob_id, run_test=None):
    prob_config = get_prob_config(phase_id, prob_id, run_test)
    org_train_data, _, _ = RawDataProcessor.get_processed_data(prob_config, "None", 0, False, False, False, False)
    cate_col = prob_config.categorical_cols
    link_data_test= '../data/captured_data/phase-3/prob-2/'
    dict_results = {}
    for col in cate_col:
        dict_true_values = (org_train_data[col].value_counts() / org_train_data.shape[0]).to_dict()
        dict_true_values = process_dict_values(dict_true_values)
        dict_non_driff = {"records": 0, "values": {key:0 for key in dict_true_values}}
        dict_driff = {"records": 0, "values": {key:0 for key in dict_true_values}}
        for index, file_path in enumerate(os.listdir(link_data_test)):
            if 'parquet' not in file_path or "123.parquet" in file_path:
                continue
            captured_data = pd.read_parquet(f'{link_data_test}{file_path}')
            dict_parquet = (captured_data[col].value_counts() / captured_data.shape[0]).to_dict()
            dict_parquet = process_dict_values(dict_parquet, list(dict_true_values.keys()))
            total_diff = compare_dict(dict_true_values, dict_parquet)
            if total_diff < 0.1:
                dict_non_driff["records"] += 1
                dict_non_driff["values"] = {key: ((value * index) + (dict_parquet[key])) / (index + 1)  for key, value in dict_non_driff["values"].items()}
            else:
                dict_driff["records"] += 1
                dict_driff["values"] = {key: ((value * index) + (dict_parquet[key])) / (index + 1)  for key, value in dict_driff["values"].items()}
        dict_results[col] = {}
        dict_results[col]['train'] = dict_true_values
        dict_results[col]['pred_drift'] = dict_driff
        dict_results[col]['pred_non_drift'] = dict_non_driff
    return dict_results

             

        