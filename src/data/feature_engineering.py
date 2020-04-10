import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import re, string
import pandas as pd


def pr(X: np.array, y: np.array, y_i: int):
    p = X[y == y_i].sum(0)
    return (p + 1) / ((y == y_i).sum() + 1)


def data_preprocessing(train_df, test_df, labels, comment):
    train_df['none'] = 1 - train_df[labels].max(axis=1)
    train_df[comment].fillna("unknown", inplace=True)
    test_df[comment].fillna("unknown", inplace=True)
    return train_df, test_df


def tokenize(s):
    re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
    return re_tok.sub(r' \1 ', s).split()


def TF_IDF_Vectorizer(train_df, test_df, comment, tokenize):
    #TODO c'est pas idéal comme fonction parce que quand on aura une phrase entrée par l'utilisateur sur streamlit,
    # il faudra simplement qu'on load le vectorizer et qu'on fasse un .transform sur cette phrase
    vec = TfidfVectorizer(ngram_range=(1, 2), tokenizer=tokenize,
                          min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
                          smooth_idf=1, sublinear_tf=1)
    trn_term_doc = vec.fit_transform(train_df[comment])
    test_term_doc = vec.transform(test_df[comment])
    return trn_term_doc, test_term_doc, vec
