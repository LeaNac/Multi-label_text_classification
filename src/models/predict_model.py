import numpy as np
import pandas as pd


def get_prediction(sentence, vectorizer, classifiers, labels):
    x_test = vectorizer.transform(pd.Series(sentence))
    preds = np.zeros((x_test.shape[0], len(labels)))
    for idx, label in enumerate(labels):
        preds[:, idx] = classifiers[label].predict_proba(x_test)[:, 1]
    return preds, x_test
