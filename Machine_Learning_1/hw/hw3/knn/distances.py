import numpy as np


def euclidean_distance(x, y):
    ans = -2 * x @ y.T
    ans += np.sum(x**2, axis=1).reshape((len(x), 1))
    ans += np.sum(y**2, axis=1).reshape((1, len(y)))
    ans = np.sqrt(ans)

    return ans



def cosine_distance(x, y):
    ans = np.zeros((len(x), len(y)))

    x_norm = np.sqrt(np.sum(x * x, axis=1)).reshape((-1, 1))
    y_norm = np.sqrt(np.sum(y * y, axis=1)).reshape((1, -1))
    pairwise_norm = x_norm @ y_norm

    ans = x @ y.T
    ans = 1 - ans / pairwise_norm
    
    return ans