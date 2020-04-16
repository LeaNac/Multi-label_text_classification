import re
import string


def preprocess_train_df(train_df, comment):
    train_df[comment].fillna("unknown", inplace=True)
    return train_df


def tokenize(s):
    re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
    return re_tok.sub(r' \1 ', s).split()
