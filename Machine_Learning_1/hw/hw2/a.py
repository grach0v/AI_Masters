# %%

import numpy as np


def get_nonzero_diag_product(X):
    diag = np.diag(X)
    ids = diag != 0
    if np.sum(ids) == 0:
        return None

    return np.prod(diag[ids])
