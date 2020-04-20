import math

import numpy as np
import pandas as pd


def get_local_weights_df(vectorizer, test_term_doc, classifiers, label):
    words = np.array(vectorizer.get_feature_names())[test_term_doc.indices]
    weights = classifiers[label].coef_.ravel()[test_term_doc.indices]
    df_words_weights = pd.DataFrame({'words': words, 'weights': weights}).sort_values(ascending=False, by='weights')
    return df_words_weights


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def odds(x):
    return sigmoid(x) / (1 - sigmoid(x))


def odds_withdraw_feature(odds_with_feature, weight_feature):
    return odds_with_feature / np.exp(weight_feature)


def odds_add_feature(odds_without_feature, weight_feature):
    return odds_without_feature * np.exp(weight_feature)


def get_global_weights_df(vectorizer, classifiers, label):
    feature_importance = classifiers[label].coef_[0]
    words = vectorizer.get_feature_names()
    df_words_weights = pd.DataFrame({'words': words, 'weights': feature_importance})
    df_words_weights = df_words_weights.sort_values(ascending=False, by='weights')
    return df_words_weights
