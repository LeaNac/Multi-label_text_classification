import pandas as pd


def load_dataset(toxic_comment_data_path):
    train_df = pd.read_csv(toxic_comment_data_path / 'train.csv')
    test_df = pd.read_csv(toxic_comment_data_path / 'test.csv')
    test_labels_df = pd.read_csv(toxic_comment_data_path / 'test_labels.csv')
    return train_df, test_df, test_labels_df
