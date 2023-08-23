from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from pandas.util import hash_pandas_object

    
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

def save_request_data(feature_df: pd.DataFrame, captured_data_dir, data_id: str):
    os.makedirs(captured_data_dir, exist_ok=True)
    if data_id.strip():
        filename = data_id
    else:
        filename = hash_pandas_object(feature_df).sum()
    output_file_path = os.path.join(captured_data_dir, f"{filename}.parquet")
    feature_df.to_parquet(output_file_path, index=False)
    return output_file_path

def handle_prediction(label_pred, proba_pred):
    res_pred = []
    for index, each in enumerate(label_pred):
        if each == "Normal" and proba_pred[index] <= 0.6:
            res_pred.append("Denial of Service")
            continue
        res_pred.append(each)
    return np.array(res_pred)
