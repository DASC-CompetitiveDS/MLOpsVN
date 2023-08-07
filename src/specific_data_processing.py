import pandas as pd
import numpy as np

def data_processing_phase2_prob2(data, phase="train"):
    drop_corr = ['feature32', 'feature36', 'feature37']
    convert_cate_cols = ['feature9', 'feature10', 'feature31', 'feature41']
    # data = data.drop(columns = drop_corr)
    for col in convert_cate_cols:
        data[col] = data[col].astype(int)
    # if phase == 'test':
    #     data["feature3"] = data["feature3"].replace({'null': '-'})
    # cate_cols = ['feature2', 'feature3', 'feature4']
    # for col in cate_cols:
    #     dict_count = data[col].value_counts().to_dict()
    #     list_below = [key for key, value in dict_count.items() if value / data.shape[0] < 0.0005]
    #     if phase == 'test':
    #         list_below.append('null')
    #     data[col] = data[col].replace({value: 'other' for value in list_below})
    return data

class ProcessData:
    HANDLE_DATA = {}
    HANDLE_DATA["phase-2_prob-1"] = data_processing_phase2_prob2
    HANDLE_DATA["phase-2_prob-2"] = data_processing_phase2_prob2