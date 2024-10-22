import json
from utils.config import AppPath
import os

class ProblemConst:
    PHASE1 = "phase-1"
    PHASE2 = "phase-2"
    PHASE3 = "phase-3"
    PROB1 = "prob-1"
    PROB2 = "prob-2"


class ProblemConfig:
    # required inputs
    phase_id: str
    prob_id: str
    test_size: float
    random_state: int

    # for original data
    raw_data_path: str
    feature_config_path: str
    category_index_path: str
    train_data_path: str
    train_x_path: str
    train_y_path: str
    test_x_path: str
    test_y_path: str

    # ml-problem properties
    target_col: str
    numerical_cols: list
    categorical_cols: list
    ml_type: str
    
    #params tuning
    params_tuning: dict
    params_fix: str

    # for data captured from API
    captured_data_dir: str
    processed_captured_data_dir: str
    # processed captured data
    captured_x_path: str
    uncertain_y_path: str
    
    #model config
    model_config_path: str


def load_feature_configs_dict(config_path: str) -> dict:
    with open(config_path) as f:
        features_config = json.load(f)
    return features_config


def create_prob_config(phase_id: str, prob_id: str, run_test=None) -> ProblemConfig:
    prob_config = ProblemConfig()
    prob_config.prob_id = prob_id
    prob_config.phase_id = phase_id
    prob_config.test_size = 0.2
    prob_config.random_state = 123

    # construct data paths for original data
    prob_config.raw_data_path = (
         AppPath.RAW_DATA_DIR / f"{phase_id}" / f"{prob_id}" / "raw_train.parquet"
    )
    prob_config.raw_data_path = f"{run_test}/{prob_config.raw_data_path}" if run_test is not None else prob_config.raw_data_path
    prob_config.external_data_path = (
        AppPath.EXTERNAL_DATA_DIR / f"{phase_id}" / f"{prob_id}" / "external_data.parquet"
    )
    prob_config.external_data_path = f"{run_test}/{prob_config.external_data_path}" if run_test is not None else prob_config.external_data_path
    prob_config.feature_config_path = (
        AppPath.RAW_DATA_DIR / f"{phase_id}" / f"{prob_id}" / "features_config.json"
    )
    prob_config.feature_config_path = f"{run_test}/{prob_config.feature_config_path}" if run_test is not None else prob_config.feature_config_path
    prob_config.train_data_path = AppPath.TRAIN_DATA_DIR / f"{phase_id}" / f"{prob_id}"
    prob_config.train_data_path = (f"{run_test}/{prob_config.train_data_path}") if run_test is not None else prob_config.train_data_path
    os.makedirs(prob_config.train_data_path, exist_ok=True)
    # prob_config.train_data_path.mkdir(parents=True, exist_ok=True)

    prob_config.category_index_path = f"{prob_config.train_data_path}/category_index.pickle"
    prob_config.category_index_path_specific_handling = f"{prob_config.train_data_path}/category_index_specific_handling.pickle"
    prob_config.dict_convert_path = f"{prob_config.train_data_path}/dict_convert.pkl"
    prob_config.train_x_path = f"{prob_config.train_data_path}/train_x.parquet"
    prob_config.train_y_path = f"{prob_config.train_data_path}/train_y.parquet"
    prob_config.train_x_drift_path = f"{prob_config.train_data_path}/train_x_drift.parquet"
    prob_config.train_y_drift_path = f"{prob_config.train_data_path}/train_y_drift.parquet"
    prob_config.test_x_path = f"{prob_config.train_data_path}/test_x.parquet"
    prob_config.test_y_path = f"{prob_config.train_data_path}/test_y.parquet"
    prob_config.test_x_drift_path = f"{prob_config.train_data_path}/test_x_drift.parquet"
    prob_config.test_y_drift_path = f"{prob_config.train_data_path}/test_y_drift.parquet"

    prob_config.add_features_model = (
        AppPath.MODEL_CONFIG_DIR / f"{phase_id}" / f"{prob_id}" / "add_features_model.yaml"
    )
    prob_config.add_features_model = f"{run_test}/{prob_config.add_features_model}" if run_test is not None else prob_config.add_features_model
    # get properties of ml-problem
    prob_config.feature_configs = load_feature_configs_dict(prob_config.feature_config_path)
    prob_config.target_col = prob_config.feature_configs.get("target_column")
    prob_config.categorical_cols = prob_config.feature_configs.get("category_columns")
    prob_config.numerical_cols = prob_config.feature_configs.get("numeric_columns")
    prob_config.ml_type = prob_config.feature_configs.get("ml_type")
    
    #create ml models params
    prob_config.params_tuning = {}
    
    prob_config.params_tuning['xgb'] = {'max_depth': ([8, 15], 'int'), 'n_estimators': (10000, 'fix'), 'subsample': ([0.6, 0.9], 'float'),
                                        'colsample_bytree': ([0.6, 0.9], 'float')}
    # 'lambda': ([1e-03, 10.0], 'log'), 'alpha': ([1e-03, 10.0], 'log')
    prob_config.params_tuning['lgbm'] = {'max_depth': ([8, 15], 'int'), 'num_leaves': ([10, 40], 'int'),
                                         'subsample': ([0.6, 0.9], 'float'), 'colsample_bytree': ([0.6, 0.9], 'float'), 
                                         'n_estimators': (10000, 'fix')}
    prob_config.params_tuning['catboost'] = {}
    prob_config.params_tuning['rdf'] = {}
    
    prob_config.params_fix = {}
    prob_config.params_fix['xgb'] = {'max_depth': 10, 'subsample': 0.8986679798510896, 'colsample_bytree': 0.8359940906525908}  
    prob_config.params_fix['lgbm'] = {'learning_rate':0.2, 'max_depth': -1, 'num_leaves': 40, 'subsample': 1, 'colsample_bytree': 0.55, 'lambda_l2': 6.3}
    prob_config.params_fix['catboost'] = {'max_depth': 12, 'n_estimators': 500}
    prob_config.params_fix['rdf'] = {}

    
    # construct data paths for API-captured data
    prob_config.captured_data_dir = (
        AppPath.CAPTURED_DATA_DIR / f"{phase_id}" / f"{prob_id}"
    )
    prob_config.captured_data_dir = f"{run_test}/{prob_config.captured_data_dir}" if run_test is not None else prob_config.captured_data_dir
    os.makedirs(prob_config.captured_data_dir, exist_ok=True)
    # prob_config.captured_data_dir.mkdir(parents=True, exist_ok=True)
    prob_config.processed_captured_data_dir = f"{prob_config.captured_data_dir}/processed"
    os.makedirs(prob_config.processed_captured_data_dir, exist_ok=True)
    # prob_config.processed_captured_data_dir.mkdir(parents=True, exist_ok=True)
    prob_config.captured_x_path = f"{prob_config.processed_captured_data_dir}/captured_x.parquet"
    prob_config.uncertain_y_path = f"{prob_config.processed_captured_data_dir}/uncertain_y.parquet"
    
    # model config path
    prob_config.model_config_path = AppPath.MODEL_CONFIG_DIR / f"{phase_id}" / f"{prob_id}" 
    prob_config.model_config_path = f"{run_test}/{prob_config.model_config_path}" if run_test is not None else prob_config.model_config_path
    os.makedirs(prob_config.model_config_path, exist_ok=True)
    # prob_config.model_config_path.mkdir(parents=True, exist_ok=True)

    return prob_config


def get_prob_config(phase_id: str, prob_id: str, run_test=None):
    prob_config = create_prob_config(phase_id, prob_id, run_test)
    return prob_config
