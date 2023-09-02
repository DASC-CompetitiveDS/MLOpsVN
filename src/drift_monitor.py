import pandas as pd
import numpy  as np
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoost, CatBoostClassifier, Pool
import argparse
from storage_utils.folder_getter import get_data
from storage_utils.get_captured_data import drop_exist
from utils.config import AppConfig
import os
from minio import Minio
from problem_config import *
import logging
import time
from glob import glob


def removed(ls, remove_item):
    ls_res = ls.copy()
    ls_res.remove(remove_item)
    return ls_res

def train_catboost(train_x, val_x, train_y, val_y, cat_features=None, early_stopping_rounds=100, verbose=False):
    train_data = Pool(train_x, train_y, cat_features=cat_features)
    val_data = Pool(val_x, val_y, cat_features=cat_features)

    cb = CatBoostClassifier(learning_rate=0.1, num_boost_round=10000, random_state=42, 
                            loss_function="Logloss", verbose=verbose, thread_count=2)

    cb.fit(train_data, eval_set=(val_data), early_stopping_rounds=early_stopping_rounds)
    return cb


def calculate_similarity_distribution(list_batch_paths, features, num_compare=5, cat_features=None, sample_size=None):
   
    sample_size = len(list_batch_paths) if sample_size is None else sample_size

    all_ref_dists = random.sample(list_batch_paths, k=min(sample_size, len(list_batch_paths)))
    all_ref_dists_idx = [i for i in range(len(all_ref_dists))]

    compare_dists_idx = [random.sample(removed(all_ref_dists_idx, idx), k=num_compare) for idx in all_ref_dists_idx]
    compare_dists_idx

    list_acc = []

    for ref_dist_idx, compare_dist_idx_set in tqdm(zip(all_ref_dists_idx, compare_dists_idx), total=len(all_ref_dists), desc="Computing pair-wise similarity"):
        for compare_dist_idx in compare_dist_idx_set:
            ref_dist = all_ref_dists[ref_dist_idx]
            compare_dist = all_ref_dists[compare_dist_idx]
            
            batch_0 = pd.read_parquet(ref_dist)
            batch_1 = pd.read_parquet(compare_dist)
            
            batch_0['label'] = 0
            batch_1['label'] = 1
            
            data = pd.concat([batch_0, batch_1])
            
            train_x, val_x, train_y, val_y = train_test_split(data[features], data['label'], test_size=0.3)
            
            cb = train_catboost(train_x, val_x, train_y, val_y, cat_features, early_stopping_rounds=100, verbose=False)
            
            acc = accuracy_score(val_y, cb.predict(val_x))
            
            list_acc.append(acc)
            
    return list_acc

def get_latest_captured_version(minio_server, paths, key_of_version_in_tags):
    minio_server = minio_server.replace('http://', '') if minio_server.startswith('http://') else minio_server
    print(minio_server)
    client = Minio(
            minio_server,
            access_key="admin",
            secret_key="password",
            secure=False
        )
        
    latest_object_path = max(paths, key=lambda path: client.stat_object('data', path).last_modified)
    latest_version = client.get_object_tags('data', latest_object_path)[key_of_version_in_tags]
    
    return latest_version    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--path", type=str, default='data')
    parser.add_argument("--phase_id", type=str, default='phase-3')
    parser.add_argument("--prob_id", type=str, default='prob-1')
    parser.add_argument("--num_compare_to_reference", type=int, default=3)
    parser.add_argument("--sample_size", type=int, default=30)    
    
    args = parser.parse_args()
    include_pattern = os.path.join(args.phase_id, args.prob_id)
    
    last_tag = None
    prob_config = create_prob_config(args.phase_id, args.prob_id)
    features = prob_config.numerical_cols + prob_config.categorical_cols
    cat_features = prob_config.categorical_cols
    
    while True:
        captured_data_paths = get_data(AppConfig().MINIO_URI, src_path='data/captured_data', include_pattern=include_pattern, return_paths=True)
        latest_tag = get_latest_captured_version(AppConfig().MINIO_URI, captured_data_paths, key_of_version_in_tags='captured_version')
        
        if latest_tag != last_tag:
            save_captured_path = os.path.join('data/captured_data', include_pattern)
            drop_exist(save_captured_path)
            get_data(AppConfig().MINIO_URI, src_path='data/captured_data', include_pattern=include_pattern, tag=('captured_version', latest_tag))
            
            similarity_scores = calculate_similarity_distribution(glob(f'{save_captured_path}/*.parquet'), features=features, cat_features=cat_features, num_compare=args.num_compare_to_reference, sample_size=args.sample_size)
            logging.info(f'Similarity distribution for version {latest_tag}:\n {similarity_scores}')
            
            last_tag = latest_tag
            
        for _ in tqdm(range(60), desc="Wait for next drift monitoring"):
            time.sleep(1)
        
        
        
