import numpy as np


def get_prediction_and_feature_importance(x_test, classifiers_and_coef, labels):
    preds = np.zeros((x_test.shape[0], len(labels)))
    feature_importance = []
    for idx, theme in enumerate(labels):
        classifier, r = classifiers_and_coef[idx][0], classifiers_and_coef[idx][1]
        preds[:, idx] = classifier.predict_proba(x_test.multiply(r))[:, 1]
        coefficients = classifier.coef_[0]
        coefficients = 100.0 * (coefficients / coefficients.max())
        feature_importance.append(coefficients)
    return preds, feature_importance
