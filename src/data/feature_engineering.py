import numpy as np
import pandas as pd


def pr(X: np.array, y: np.array, y_i: int):
    p = X[y == y_i].sum(0)
    return (p + 1) / ((y == y_i).sum() + 1)
