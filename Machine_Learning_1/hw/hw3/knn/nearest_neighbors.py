import numpy as np

from knn.distances import euclidean_distance, cosine_distance


def get_best_ranks(ranks, top, axis=1, return_ranks=False):
    raise NotImplementedError()


class NearestNeighborsFinder:
    def __init__(self, n_neighbors, metric="euclidean"):
        self.n_neighbors = n_neighbors

        if metric == "euclidean":
            self._metric_func = euclidean_distance
        elif metric == "cosine":
            self._metric_func = cosine_distance
        else:
            raise ValueError("Metric is not supported", metric)
        self.metric = metric

    def fit(self, X, y=None):
        self._X = X
        return self

    def kneighbors(self, X, return_distance=False):
        real_distances = self._metric_func(X, self._X)
        
        k = self.n_neighbors
        indexes = np.argpartition(real_distances, min(k, len(self._X) - 1), axis=1)[:, :k]
        real_distances = np.take_along_axis(real_distances, indexes, axis=1)

        sorted_indexes = np.argsort(real_distances, axis=1)

        indices = np.take_along_axis(indexes, sorted_indexes, axis=1)
        if return_distance:
            distances = np.take_along_axis(real_distances, sorted_indexes, axis=1)
            return distances, indices
        else:
            return indices



