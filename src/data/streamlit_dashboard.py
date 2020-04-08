import streamlit as st
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import plotly.express as px

TOXIC_COMMENT_DATA_PATH = Path(__file__).parent.parent.parent / 'data/'


@st.cache
def load_dataset(toxic_comment_data_path):
    train_df = pd.read_csv(toxic_comment_data_path / 'train.csv')
    test_df = pd.read_csv(toxic_comment_data_path / 'test.csv')
    test_labels_df = pd.read_csv(toxic_comment_data_path / 'test_labels.csv')
    return train_df, test_df, test_labels_df


train_df, test_df, test_labels_df = load_dataset(TOXIC_COMMENT_DATA_PATH)
lens_df = train_df.comment_text.str.len()
lens_df.mean(), lens_df.std(), lens_df.max()
st.write(lens_df)

st.title('Data exploration')
st.subheader('Comments Length Histogram')

hist = px.histogram(data_frame=lens_df, x='comment_text')
st.write(hist)
