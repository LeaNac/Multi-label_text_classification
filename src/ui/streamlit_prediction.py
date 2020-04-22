import glob

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.conf.global_variables import MODELS_PATH, LABELS
from src.models.models_interpretation import get_local_weights_df, get_global_weights_df
from src.models.models_predict import get_prediction
from src.models.models_save_and_load import load_models

models_filenames = glob.glob(str(MODELS_PATH / 'model_*.pkl'))
vectorizer_filename = MODELS_PATH / 'vectorizer.pkl'

st.title("Catégorisation d'une phrase toxique")
classifiers, vectorizer = load_models(models_filenames, vectorizer_filename)

input_sentence = st.text_input('Entrez une phrase en anglais, le modèle étant préentraîné sur le dataset Toxic Comment:')
st.markdown("La probabilité donnée par chaque régression logistique donne des pourcentages d'appartenance à chaque classe. Il y a possibilité d'afficher les odds pour mieux comprendre comment les poids font évoluer la prédiction.")
st.latex(r'odds=\frac{P(y=1)}{P(y=0)}')
preds, test_term_doc = get_prediction(input_sentence, vectorizer, classifiers, LABELS)
preds_df = pd.DataFrame({'label': LABELS, 'preds': preds[0] * 100})
odds_df = pd.DataFrame({'label': LABELS, 'odds': preds[0] / (1 - preds[0])})
is_odds = st.checkbox('Afficher les odds', key='is_odds')
if not is_odds:
    fig = px.bar(preds_df,
                 x='preds',
                 y='label',
                 color='preds',
                 range_x=[0, 100],
                 orientation='h',
                 range_color=[0, 100],
                 color_continuous_scale='Reds',
                 title="Probabilité d'appartenance au label")
else:
    fig = px.bar(odds_df,
                 x='odds',
                 y='label',
                 color='odds',
                 orientation='h',
                 color_continuous_scale='Reds',
                 title="Odds d'appartenance au label")

predictions_chart = st.plotly_chart(fig)

st.title('Interprétation du modèle')

chosen_label = st.sidebar.selectbox(
    "Catégorie à afficher pour l'interprétation",
    LABELS
)

st.subheader(f'Interprétation locale pour la catégorie {chosen_label}')
st.markdown(
    f"Nous voyons ici au niveau d'une phrase quels mots ont le plus contribué à la prédiction dans la catégorie {chosen_label}. Pour cela, nous affichons les poids de la régression logistique ; il y a aussi la possibilité d'afficher leur exponentielle.")
df_local_weights_for_label = get_local_weights_df(vectorizer, test_term_doc, classifiers, chosen_label)
df_global_weights_for_label = get_global_weights_df(vectorizer, classifiers, chosen_label)
df_local_weights_for_label['exp_weights'] = np.exp(df_local_weights_for_label['weights'])
df_global_weights_for_label['exp_weights'] = np.exp(df_global_weights_for_label['weights'])
max_weight_for_label = df_global_weights_for_label['weights'].max()
max_exp_weight_for_label = df_global_weights_for_label['exp_weights'].max()
are_weights_exponential_local = st.checkbox('Afficher les poids exponentiels', key='are_weights_exponential_local')

if not are_weights_exponential_local:
    fig2 = px.bar(df_local_weights_for_label,
                  x='words',
                  y='weights',
                  color='weights',
                  orientation='v',
                  range_y=[0, max_weight_for_label+0.1*max_weight_for_label],
                  range_color=[0, max_weight_for_label],
                  color_continuous_scale='Reds',
                  title="Weights")
else:
    fig2 = px.bar(df_local_weights_for_label,
                  x='words',
                  y='exp_weights',
                  color='exp_weights',
                  orientation='v',
                  range_y=[0, max_exp_weight_for_label+0.1*max_exp_weight_for_label],
                  range_color=[0, max_exp_weight_for_label],
                  color_continuous_scale='Reds',
                  title="Exp(weights)")

words_weights_chart = st.plotly_chart(fig2)

st.subheader(f'Interprétation globale pour la catégorie {chosen_label}')
st.markdown(
    f"Nous voyons au niveau du classifieur quels mots sont les plus impactants pour la classification dans la catégorie {chosen_label}. Il y a aussi la possibilité d'afficher leur exponentielle.")
are_weights_exponential_global = st.checkbox('Afficher les poids exponentiels', key='are_weights_exponential_global')

fig3 = go.Figure()
n_words = st.slider('Nombre de mots à afficher', min_value=5, max_value=20, value=5, step=1)
df_global_weights_n_words = df_global_weights_for_label.head(n_words)
if not are_weights_exponential_global:
    fig3.add_trace(go.Bar(
        x=df_global_weights_n_words['words'],
        y=df_global_weights_n_words['weights'],
        name='weights',
        marker_color='lightsalmon'
    ))
else:
    fig3.add_trace(go.Bar(
        x=df_global_weights_n_words['words'],
        y=np.exp(df_global_weights_n_words['weights']),
        name='weights',
        marker_color='indianred'
    ))
words_global_weights_chart = st.plotly_chart(fig3)
