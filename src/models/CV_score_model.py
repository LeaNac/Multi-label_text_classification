import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from src.data.feature_engineering import pr


def get_mdl_cross_val_score(X, y):
    r = np.log(pr(X, 1, y) / pr(X, 0, y))
    classifier = LogisticRegression()
    X_nb = X.multiply(r)
    cv_score = np.mean(cross_val_score(classifier, X_nb, y, cv=3, scoring='roc_auc'))
    return cv_score


def compute_CV_score_for_each_class(df, labels, term_doc):
    scores = []
    for label in labels:
        target = df[label].values
        X = term_doc

        cv_score = get_mdl_cross_val_score(X, target)
        scores.append(cv_score)
        print('CV score for class {} is {}'.format(label, cv_score))
    return scores