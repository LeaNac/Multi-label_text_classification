import glob
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append('.')
from src.models.models_interpretation import get_local_weights_df, get_global_weights_df
from src.models.models_predict import get_prediction
from src.conf.global_variables import MODELS_PATH, LABELS
from src.models.models_save_and_load import load_models

models_filenames = glob.glob(str(MODELS_PATH / 'model_*.pkl'))
vectorizer_filename = MODELS_PATH / 'vectorizer.pkl'

st.title("Catégorisation d'une phrase toxique")
classifiers, vectorizer = load_models(models_filenames, vectorizer_filename)

input_sentence = st.text_input('Entrez une phrase:')
preds, test_term_doc = get_prediction(input_sentence, vectorizer, classifiers, LABELS)
preds_df = pd.DataFrame({'label': LABELS, 'preds': preds[0] * 100})
fig = px.bar(preds_df,
             x='preds',
             y='label',
             color='preds',
             range_x=[0, 100],
             orientation='h',
             # height=400,
             range_color=[0, 100],
             color_continuous_scale='Reds',
             title="Probabilité d'appartenance au label")
predictions_chart = st.plotly_chart(fig)

st.title('Interprétation du modèle')

chosen_label = st.sidebar.selectbox(
    "Catégorie à afficher pour l'interprétation",
    LABELS
)


st.subheader('Interprétation de la phrase')
df_words_weight_toxic = get_local_weights_df(vectorizer, test_term_doc, classifiers, chosen_label)
fig2 = px.bar(df_words_weight_toxic,
              x='words',
              y='weights')
words_weights_chart = st.plotly_chart(fig2)

st.subheader('Interprétation globale (mots les plus $#@*)')
df_global_weights = get_global_weights_df(vectorizer, classifiers, chosen_label)
fig3 = px.bar(df_global_weights.head(10),
              x='words',
              y='weights')
words_global_weights_chart = st.plotly_chart(fig3)
