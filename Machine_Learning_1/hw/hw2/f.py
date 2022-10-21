# %%

import numpy as np


def encode_rle(x):
    ids = np.concatenate([[False], x[1:] == x[:-1], [False]])
    cum_ids = np.cumsum(ids)
    cum_ids = cum_ids[~ids]

    return x[~ids[:-1]], cum_ids[1:] - cum_ids[:-1] + 1
