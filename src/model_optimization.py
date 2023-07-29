import numpy as np
from catboost import CatBoostRegressor, CatBoostClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, f1_score, roc_auc_score, accuracy_score
import optuna
from optuna.samplers import TPESampler
import logging
import warnings
import mlflow
import lightgbm as lgb
from optuna.integration.lightgbm import LightGBMTunerCV
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype

warnings.filterwarnings("ignore")


def objective(trial, train_data, valid_data, type_model, task, params_tuning, cat_features, is_class_weight, model_name):
    mlflow.start_run(run_name=f"{model_name}_tuning_{trial.number}")

    X_train, X_valid, y_train, y_valid = train_data[0], valid_data[0], train_data[1], valid_data[1]
    unique_n = len(np.unique(y_train))
    param_grid = {}
    for key, value in params_tuning.items():
        if value[1] == 'float':
            param_grid[key] = trial.suggest_float(key, value[0][0], value[0][1])
        elif value[1] == 'int':
            param_grid[key] = trial.suggest_int(key, value[0][0], value[0][1])
        elif value[1] == 'fix':
            param_grid[key] = value[0]

    # logging.info(param_grid)
    if is_class_weight is True:
        param_grid['class_weight'] = 'balanced'
        
    if type_model == 'xgb':
        reg = XGBRegressor(**param_grid, verbose=0) if task == 'reg' else \
              XGBClassifier(objective= "binary:logistic" if unique_n == 2 else "multi:softprob", **param_grid, verbose=0)
        reg.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=0, early_stopping_rounds=300)
    elif type_model == 'lgbm':
        reg = LGBMRegressor(metric=None, **param_grid, verbose=0) if task == 'reg' else \
              LGBMClassifier(metric=None, **param_grid, verbose=0)
        eval_metric = "binary_logloss" if unique_n == 2 else "multi_logloss"
        eval_metric = eval_metric if task == 'clf' else "rmse"
        reg.fit(X_train, y_train, eval_set=(X_valid, y_valid), eval_metric=eval_metric, 
                categorical_feature=cat_features, verbose=0, early_stopping_rounds=300)        
    elif type_model == 'catboost':
        pass
    else:
        pass
    
    if type_model == 'reg':
        key_metrics = 'mean_squared_error'
        res = mean_squared_error(y_valid, reg.predict(X_valid), squared=False)
    else:
        key_metrics = 'roc_auc_score' if unique_n == 2 else 'f1_score'
        res = roc_auc_score(y_valid, reg.predict_proba(X_valid)[:, 1]) if unique_n == 2 else \
              accuracy_score(y_valid, reg.predict(X_valid))
    # with mlflow.start_run(run_name=f"{model_name}_tuning_{trial.trial_id}"):
    mlflow.set_tag('type_model', type_model)
    mlflow.log_params(reg.get_params())
    mlflow.log_metrics({key_metrics: res})  
    mlflow.log_metrics({'best_interation_':reg.best_iteration_})
    mlflow.end_run()
    
    return res

def objective_crossvalidation(trial, train_data, valid_data, type_model, task, params_tuning, cat_features, is_class_weight, model_name):
    mlflow.start_run(run_name=f"{model_name}_tuning_{trial.number}")

    X_train, X_valid, y_train, y_valid = train_data[0], valid_data[0], train_data[1], valid_data[1]
    unique_n = len(np.unique(y_train))
    param_grid = {}
    for key, value in params_tuning.items():
        if value[1] == 'float':
            param_grid[key] = trial.suggest_float(key, value[0][0], value[0][1])
        elif value[1] == 'int':
            param_grid[key] = trial.suggest_int(key, value[0][0], value[0][1])
        elif value[1] == 'fix':
            param_grid[key] = value[0]

    # logging.info(param_grid)
    if is_class_weight is True:
        param_grid['class_weight'] = 'balanced'
        
    if type_model == 'xgb':
        pass
    elif type_model == 'lgbm':
        # reg = LGBMRegressor(metric=None, **param_grid, verbose=0) if type_model == 'reg' else \
        #       LGBMClassifier(metric=None, **param_grid, verbose=0)
        # eval_metric = "binary_logloss" if unique_n == 2 else "multi_logloss"
        # eval_metric = eval_metric if task == 'clf' else "rmse"
        # reg.fit(X_train, y_train, eval_set=(X_valid, y_valid), eval_metric=eval_metric, 
        #         categorical_feature=cat_features, verbose=0, early_stopping_rounds=300)   
        
        lgb.cv(param_grid)     
    elif type_model == 'catboost':
        pass
    else:
        pass
    
    if type_model == 'reg':
        key_metrics = 'mean_squared_error'
        res = mean_squared_error(y_valid, reg.predict(X_valid), squared=False)
    else:
        key_metrics = 'roc_auc_score' if unique_n == 2 else 'f1_score'
        res = roc_auc_score(y_valid, reg.predict_proba(X_valid)[:, 1]) if unique_n == 2 else \
              accuracy_score(y_valid, reg.predict(X_valid))
    # with mlflow.start_run(run_name=f"{model_name}_tuning_{trial.trial_id}"):
    mlflow.set_tag('type_model', type_model)
    mlflow.log_params(reg.get_params())
    mlflow.log_metrics({key_metrics: res})  
    mlflow.log_metrics({'best_interation_':reg.best_iteration_})
    mlflow.end_run()
    
    return res

def get_best_params(train_data, valid_data, type_model, task, params_tuning, cat_features, is_class_weight, train_time, idx_phase = "1_1", model_name=None, args=None):
    direction = "minimize" if task == 'reg' else "maximize"
    func = lambda trial: objective(trial, train_data, valid_data, type_model, task, params_tuning, cat_features, is_class_weight, model_name)
    study_name_save = f"{idx_phase}_{type_model}_{'' if is_class_weight is False else 'classweight'}_{train_time}"
    study = optuna.create_study(direction=direction, sampler=TPESampler(), study_name=study_name_save)
    study.optimize(func, timeout=train_time)
    logging.info(f'Number of finished trials: {len(study.trials)}')
    trial = study.best_trial
    logging.info('Best trial:')
    logging.info(f'** Value: {trial.value}')
    logging.info(f'** Params: {trial.params}')
    params = trial.params
    return params

def get_best_params_cv(train_data, type_model, task, cat_features, is_class_weight, train_time, idx_phase = "1_1", model_name=None, args=None):
    direction = "minimize" if task == 'reg' else "maximize"
    # func = lambda trial: objective(trial, train_data, valid_data, type_model, task, params_tuning, cat_features, is_class_weight, model_name)
    # study_name_save = f"{idx_phase}_{type_model}_{'' if is_class_weight is False else 'classweight'}_{train_time}"
    # study = optuna.create_study(direction=direction, sampler=TPESampler(), study_name=study_name_save)
    # study.optimize(func, timeout=train_time)
    # logging.info(f'Number of finished trials: {len(study.trials)}')
    # trial = study.best_trial
    # logging.info('Best trial:')
    # logging.info(f'** Value: {trial.value}')
    X_train, y_train = train_data
    unique_n = len(np.unique(y_train))
    
    if type_model == 'lgbm':
        if not is_numeric_dtype(y_train):
            y_train = y_train.astype("category").cat.codes

        dtrain = lgb.Dataset(X_train, label=y_train)
        eval_metric = "binary_logloss" if unique_n == 2 else "multi_logloss"
        eval_metric = eval_metric if task == 'clf' else "rmse"
        
        objective = "binary" if unique_n == 2 else "multiclass"
        objective = objective if task == 'clf' else "regression"
        # num_class = unique_n if objective is 'multiclass' else 1

        params = {
        "objective": objective,
        "metric": eval_metric,
        "verbosity": -1,
        "boosting_type": "gbdt",
        "learning_rate": args.learning_rate
        }
        
        if objective == 'multiclass':
            params.update({'num_class': unique_n})

        tuner = LightGBMTunerCV(
            time_budget=train_time, params=params,
            train_set=dtrain, categorical_feature=cat_features,
            nfold=5, stratified=True,
            num_boost_round=10000, early_stopping_rounds=300,
            return_cvbooster=True, optuna_seed = 123, seed = 123,
        )
        
        tuner.run()
        num_boost_round = tuner.get_best_booster().best_iteration
        params = tuner.best_params      
    else:
        pass
    
    logging.info(f'** Params: {params}')
    logging.info(f'** Best iteration: {num_boost_round}')
    return params, num_boost_round

def model_training(train_data, valid_data, type_model, task, param_grid, cat_features, is_class_weight, n_estimators=None):
    logging.info("=============== TRAINING =============")
    X_train, X_valid, y_train, y_valid = train_data[0], valid_data[0], train_data[1], valid_data[1]
    unique_n = len(np.unique(y_train))
    
    n_estimators = 10000 if n_estimators is None else n_estimators
    early_stopping_rounds = 0 if n_estimators is None else 300
    if 'metric' in param_grid.keys():
        param_grid.pop('metric')
    
    if is_class_weight is True:
        param_grid['class_weight'] = 'balanced'
    if type_model == 'xgb':
        reg = XGBRegressor(**param_grid, verbose=0, n_estimators=n_estimators) if type_model == 'reg' else \
              XGBClassifier(objective= "binary:logistic" if unique_n == 2 else "multi:softprob", **param_grid, verbose=0, n_estimators=n_estimators)
        reg.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=0, early_stopping_rounds=early_stopping_rounds)
    elif type_model == 'lgbm':
        reg = LGBMRegressor(metric=None, **param_grid, n_estimators=n_estimators, verbose=0) if type_model == 'reg' else \
              LGBMClassifier(metric=None, **param_grid, n_estimators=n_estimators, verbose=0)
        eval_metric = "binary_logloss" if unique_n == 2 else "multi_logloss"
        eval_metric = eval_metric if task == 'clf' else "rmse"
        reg.fit(X_train, y_train, eval_set=(X_valid, y_valid), eval_metric=eval_metric, 
                categorical_feature=cat_features, verbose=0, early_stopping_rounds=early_stopping_rounds)        
    elif type_model == 'catboost':
        reg = None
    else:
        reg = RandomForestRegressor(**param_grid, verbose=0) if type_model == 'reg' else \
              RandomForestClassifier(**param_grid, verbose=0)
        reg.fit(X_train, y_train)
    if type_model == 'reg':
        pred = reg.predict(X_valid)
        res = mean_squared_error(y_valid, pred, squared=False)
    else:
        pred = reg.predict_proba(X_valid)[:, 1] if unique_n == 2 else reg.predict(X_valid)
        res = roc_auc_score(y_valid, pred) if unique_n == 2 else accuracy_score(y_valid, pred)
    return reg, res, pred
