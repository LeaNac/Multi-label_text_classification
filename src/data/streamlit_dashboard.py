import streamlit as st
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

TOXIC_COMMENT_DATA_PATH = Path(__file__).parent.parent.parent / 'dataset/'


@st.cache
def load_dataset(toxic_comment_data_path):
    train_df = pd.read_csv(toxic_comment_data_path / 'train.csv')
    test_df = pd.read_csv(toxic_comment_data_path / 'test.csv')
    test_labels_df = pd.read_csv(toxic_comment_data_path / 'test_labels.csv')
    return train_df, test_df, test_labels_df


train_df, test_df, test_labels_df = load_dataset(TOXIC_COMMENT_DATA_PATH)
lens = train_df.comment_text.str.len()
lens.mean(), lens.std(), lens.max()
st.write(lens)

st.title('Data exploration')
st.subheader('Comments Length Histogram')

fig1 = lens.hist()
hist = st.write(fig1)
