import pandas as pd
import pytest


@pytest.fixture
def dataset_1():
    df = pd.DataFrame({'id': [0],
                       'comment_text': ['This is an example sentence'],
                       'toxic': [0],
                       'severe_toxic': [0],
                       'obscene': [0],
                       'threat': [0],
                       'insult': [0],
                       'identity_hate': [0]})
    return df
