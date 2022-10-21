# %%

import numpy as np


def calc_expectations(h, w, X, Q):
    Q = np.block([
        [np.zeros((h - 1, w - 1)), np.zeros((h - 1, Q.shape[1]))],
        [np.zeros((Q.shape[0], w - 1)), Q]
    ])

    conv = np.ones((h, w))
    view_shape = tuple(np.subtract(Q.shape, conv.shape) + 1) + conv.shape
    strides = Q.strides + Q.strides

    sub_matrices = np.lib.stride_tricks.as_strided(Q, view_shape, strides)

    probs = (sub_matrices * conv).sum(axis=(2, 3))

    return X * probs
