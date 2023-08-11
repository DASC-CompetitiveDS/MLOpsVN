import logging
import os
from pathlib import Path
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class AppPath:
    ROOT_DIR = Path(".")
    DATA_DIR = ROOT_DIR / "data"
    # store raw data
    RAW_DATA_DIR = DATA_DIR / "raw_data"
    #store external data
    EXTERNAL_DATA_DIR = DATA_DIR / "external_data"
    # store preprocessed training data
    TRAIN_DATA_DIR = DATA_DIR / "train_data"
    # store configs for deployments
    MODEL_CONFIG_DIR = DATA_DIR / "model_config"
    # store captured data
    CAPTURED_DATA_DIR = DATA_DIR / "captured_data"


AppPath.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
AppPath.TRAIN_DATA_DIR.mkdir(parents=True, exist_ok=True)
AppPath.MODEL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
AppPath.CAPTURED_DATA_DIR.mkdir(parents=True, exist_ok=True)


class AppConfig:
    # MLFLOW_TRACKING_URI = 'http://localhost:5000'
    MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
    MLFLOW_MODEL_PREFIX = "model"
    
def get_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure()
    sns.heatmap(cm,
                annot=True,
                fmt='g')
    plt.ylabel('Prediction',fontsize=13)
    plt.xlabel('Actual',fontsize=13)
    plt.title('Confusion Matrix',fontsize=17)
    # plt.show()
    return fig

def get_feature_importance(model, importance_type='split', model_type='lgbm'):
    if model_type == 'lgbm' and importance_type in ['split', 'gain']:
        importance_df = pd.DataFrame({'Feature':model.feature_name_,'Importance':model.booster_.feature_importance(importance_type)})
    else:
        pass
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    fig = plt.figure(figsize=(15,10))
    sns.barplot(importance_df, x='Importance', y='Feature')
    plt.title(f'feature importances', fontsize=17)
    return fig, importance_df.to_dict('records')

    
