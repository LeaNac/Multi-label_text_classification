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


