import numpy as np
from catboost import CatBoostRegressor, CatBoostClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, f1_score, roc_auc_score
import optuna
from optuna.samplers import TPESampler
import logging
import warnings
warnings.filterwarnings("ignore")


def objective(trial, train_data, valid_data, type_model, task, params_tuning, cat_features, is_class_weight):
    X_train, X_valid, y_train, y_valid = train_data[0], valid_data[0], train_data[1], valid_data[1]
    unique_n = len(np.unique(y_train))
    param_grid = {}
    for key, value in params_tuning.items():
        if value[1] == 'float':
            param_grid[key] = trial.suggest_float(key, value[0][0], value[0][1])
        elif value[1] == 'int':
            param_grid[key] = trial.suggest_int(key, value[0][0], value[0][1])
    # logging.info(param_grid)
    if is_class_weight is True:
        param_grid['class_weight'] = 'balanced'
        
    if type_model == 'xgb':
        reg = XGBRegressor(**param_grid, verbose=0) if type_model == 'reg' else \
              XGBClassifier(objective= "binary:logistic" if unique_n == 2 else "multi:softprob", **param_grid, verbose=0)
        reg.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], early_stopping_rounds=300, verbose=0)
    elif type_model == 'lgbm':
        reg = LGBMRegressor(metric=None, **param_grid, verbose=0) if type_model == 'reg' else \
              LGBMClassifier(metric=None, **param_grid, verbose=0)
        eval_metric = "binary_logloss" if unique_n == 2 else "multi_logloss"
        eval_metric = eval_metric if task == 'clf' else "rmse"
        reg.fit(X_train, y_train, eval_set=(X_valid, y_valid), eval_metric=eval_metric, 
                categorical_feature=cat_features, early_stopping_rounds=300, verbose=0)        
    elif type_model == 'catboost':
        pass
    else:
        pass
    
    if type_model == 'reg':
        res = mean_squared_error(y_valid, reg.predict(X_valid), squared=False)
    else:
        res = roc_auc_score(y_valid, reg.predict_proba(X_valid)[:, 1]) if unique_n == 2 else \
              roc_auc_score(y_valid, reg.predict_proba(X_valid), multi_class='ovr')
    return res


def get_best_params(train_data, valid_data, type_model, task, params_tuning, cat_features, is_class_weight, train_time, idx_phase = "1_1"):
    direction = "minimize" if task == 'reg' else "maximize"
    func = lambda trial: objective(trial, train_data, valid_data, type_model, task, params_tuning, cat_features, is_class_weight)
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

def model_training(train_data, valid_data, type_model, task, param_grid, cat_features, is_class_weight):
    X_train, X_valid, y_train, y_valid = train_data[0], valid_data[0], train_data[1], valid_data[1]
    unique_n = len(np.unique(y_train))
    if is_class_weight is True:
        param_grid['class_weight'] = 'balanced'
    if type_model == 'xgb':
        reg = XGBRegressor(**param_grid, verbose=0) if type_model == 'reg' else \
              XGBClassifier(objective= "binary:logistic" if unique_n == 2 else "multi:softprob", **param_grid, verbose=0)
        reg.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], early_stopping_rounds=300, verbose=0)
    elif type_model == 'lgbm':
        reg = LGBMRegressor(metric=None, **param_grid, verbose=0) if type_model == 'reg' else \
              LGBMClassifier(metric=None, **param_grid, verbose=0)
        eval_metric = "binary_logloss" if unique_n == 2 else "multi_logloss"
        eval_metric = eval_metric if task == 'clf' else "rmse"
        reg.fit(X_train, y_train, eval_set=(X_valid, y_valid), eval_metric=eval_metric, 
                categorical_feature=cat_features, early_stopping_rounds=300, verbose=0)        
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
        pred = reg.predict_proba(X_valid)[:, 1] if unique_n == 2 else reg.predict_proba(X_valid)
        res = roc_auc_score(y_valid, pred) if unique_n == 2 else roc_auc_score(y_valid, pred, multi_class='ovr')
    return reg, res, pred
