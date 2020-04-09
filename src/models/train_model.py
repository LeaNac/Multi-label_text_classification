import numpy as np
from sklearn.linear_model import LogisticRegression
from src.data.feature_engineering import pr


def get_fit_mdl(x, y):
    y = y.values
    r = np.log(pr(x, 1, y) / pr(x, 0, y))
    m = LogisticRegression()
    x_nb = x.multiply(r)
    return m.fit(x_nb, y), r


def fit_classifier_for_each_theme_and_get_its_naiveB_coef(df, train_term_doc, labels):
    all_classifiers_and_r = []
    for idx, theme in enumerate(labels):
        print('fit', theme)
        target = df[theme]
        classifier, r = get_fit_mdl(train_term_doc, target)
        all_classifiers_and_r.append([classifier, r])
    return all_classifiers_and_r


def get_feature_importance_LR(labels, classifiers_and_coef):
    feature_importance = []
    for idx, theme in enumerate(labels):
        classifier = classifiers_and_coef[idx][0]
        coefficients = classifier.coef_[0]
        coefficients = 100.0 * (coefficients / coefficients.max())
        feature_importance.append(coefficients)
    return feature_importance


def get_feature_importance_LR_fois_nb_coef(labels, classifiers_and_coef):
    feature_importance = []
    nb_de_coef = classifiers_and_coef[0][1].shape[1]
    for idx, theme in enumerate(labels):
        classifier_weigths = classifiers_and_coef[idx][0].coef_[0].reshape((nb_de_coef, 1))
        r = classifiers_and_coef[idx][1].reshape((nb_de_coef, 1))
        weights_and_r = np.concatenate((classifier_weigths, r), axis=1)
        weights_and_r = np.array(weights_and_r)
        weights_and_r_multiplication = [weights_and_r[idx][0] * weights_and_r[idx][1] for idx in range(nb_de_coef)]
        weights_and_r_multiplication = 100.0 * (weights_and_r_multiplication / max(weights_and_r_multiplication))
        feature_importance.append(weights_and_r_multiplication)
    return feature_importance
