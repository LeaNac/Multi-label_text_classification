import numpy as np
import pandas as pd


def get_prediction(sentence, vectorizer, classifiers, labels):
    x_test = vectorizer.transform(pd.Series(sentence))
    preds = np.zeros((x_test.shape[0], len(labels)))
    if not is_vectorized_sentence_empty(x_test):
        for idx, label in enumerate(labels):
            preds[:, idx] = classifiers[label].predict_proba(x_test)[:, 1]
    return preds, x_test


def is_vectorized_sentence_empty(vectorized_sentence):
    return vectorized_sentence.nnz == 0
