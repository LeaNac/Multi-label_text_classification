from pathlib import Path
import pandas as pd

TOXIC_COMMENT_DATA_PATH = Path().parent.parent.parent / 'data/'


def load_dataset(toxic_comment_data_path):
    train_df = pd.read_csv(toxic_comment_data_path / 'train.csv')
    test_df = pd.read_csv(toxic_comment_data_path / 'test.csv')
    test_labels_df = pd.read_csv(toxic_comment_data_path / 'test_labels.csv')
    return train_df, test_df, test_labels_df
