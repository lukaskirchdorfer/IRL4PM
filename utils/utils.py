import pandas as pd
from typing import List

def split_log(log: pd.DataFrame, train_ratio: float = 0.8):
    """
    Split the log chronologically into training and testing sets.
    """
    log = log.sort_values('start_timestamp')
    train_log = log.iloc[:int(len(log) * train_ratio)]
    test_log = log.iloc[int(len(log) * train_ratio):]
    return train_log, test_log

def split_log_train_val_test(log: pd.DataFrame, train_ratio: float = 0.64, val_ratio: float = 0.16, test_ratio: float = 0.2):
    """
    Split the log into training, validation, and testing sets.
    """
    log = log.sort_values('start_timestamp')
    train_log = log.iloc[:int(len(log) * train_ratio)]
    val_log = log.iloc[int(len(log) * train_ratio):int(len(log) * (train_ratio + val_ratio))]
    test_log = log.iloc[int(len(log) * (train_ratio + val_ratio)):]
    return train_log, val_log, test_log

def split_log_random(log: pd.DataFrame, train_ratio: float = 0.8):
    """
    Split the log randomly into training and testing sets.
    """
    log = log.sample(frac=1).reset_index(drop=True)
    train_log = log.iloc[:int(len(log) * train_ratio)]
    test_log = log.iloc[int(len(log) * train_ratio):]
    return train_log, test_log


def get_unique_values_for_categorical_features(log: pd.DataFrame, categorical_features: List[str]):
    cat_features_dict = {}
    for cat_feature in categorical_features:
        cat_features_dict[cat_feature] = list(log[cat_feature].unique())
    return cat_features_dict
