# %%

import numpy as np

# %%


def select(x, ind):
    return x[np.arange(x.shape[0])[:, None], ind]


def get_best_indices(ranks: np.ndarray, top: int, **params) -> np.ndarray:
    axis = params.get("axis", 1)
    if len(ranks.shape) > 2:
        return np.argpartition(-ranks, range(0, top), axis=axis).take(
            indices=range(0, top), axis=axis
        )
    max_ind = np.argpartition(ranks, -top, axis=axis)[:, -top:]
    max_val = select(ranks, max_ind)
    return select(max_ind, np.fliplr(np.argsort(max_val)))


if __name__ == "__main__":

    with open('input.bin', 'rb') as f_data:
        ranks = np.load(f_data)

    indices = get_best_indices(ranks, 5)

    with open('output.bin', 'wb') as f_data:
        np.save(f_data, indices)
