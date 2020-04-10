import numpy as np


def get_prediction(x_test, classifiers, labels):
    preds = np.zeros((x_test.shape[0], len(labels)))
    for idx, label in enumerate(labels):
        preds[:, idx] = classifiers[label].predict_proba(x_test)[:, 1]
    return preds
