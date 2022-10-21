# %%

import numpy as np


def replace_nan_to_means(X):
    X = X.copy()

    nan_ids = np.isnan(X)
    X[nan_ids] = 0

    sums = X.sum(axis=0)

    n_values = X.shape[0] - nan_ids.sum(axis=0)
    n_values[n_values == 0] = 1

    mean_values = sums / n_values

    inds = np.where(nan_ids)

    X[inds] = np.take(mean_values, inds[1])

    return X
