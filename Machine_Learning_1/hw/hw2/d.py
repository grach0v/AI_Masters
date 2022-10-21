# %%

import numpy as np


def get_max_after_zero(x):
    ids = x == 0
    x = x[1:][ids[:-1]]

    if len(x) > 0:
        return np.max(x)
    else:
        return None
