import numpy as np
import pandas as pd
from src.conf.global_variables import COMMENT, LABELS


def pr(X: np.array, y: np.array, y_i: int):
    p = X[y == y_i].sum(0)
    return (p + 1) / ((y == y_i).sum() + 1)


def data_preprocessing(train_df, test_df, labels, comment):
    train_df['none'] = 1 - train_df[labels].max(axis=1)
    train_df[comment].fillna("unknown", inplace=True)
    test_df[comment].fillna("unknown", inplace=True)
    return train_df, test_df
