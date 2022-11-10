import numpy as np

from sklearn.neighbors import NearestNeighbors
from knn.nearest_neighbors import NearestNeighborsFinder
from scipy import stats as ss


class KNNClassifier:
    EPS = 1e-5

    def __init__(self, n_neighbors, algorithm='my_own', metric='euclidean', weights='uniform'):
        if algorithm == 'my_own':
            finder = NearestNeighborsFinder(n_neighbors=n_neighbors, metric=metric)
        elif algorithm in ('brute', 'ball_tree', 'kd_tree',):
            finder = NearestNeighbors(n_neighbors=n_neighbors, algorithm=algorithm, metric=metric)
        else:
            raise ValueError("Algorithm is not supported", metric)

        if weights not in ('uniform', 'distance'):
            raise ValueError("Weighted algorithm is not supported", weights)

        self._finder = finder
        self._weights = weights

    def fit(self, X, y=None):
        self._finder.fit(X)
        self._labels = np.asarray(y)
        return self

    def _predict_precomputed(self, indices, distances):
        if self._weights == 'uniform':
            return ss.mode(self._labels[indices], axis=1).mode.reshape(-1)

        elif self._weights == 'distance':
            unique_labels = np.unique(self._labels)
            weighted_labels = np.zeros((len(distances), len(unique_labels)))

            for label_i, label in enumerate(unique_labels):
                weighted_labels[:, label_i] = \
                    ((self._labels[indices] == label) / (distances + 1e-5)).sum(axis=1)

            return unique_labels[weighted_labels.argmax(axis=1)]

    def kneighbors(self, X, return_distance=False):
        return self._finder.kneighbors(X, return_distance=return_distance)

    def predict(self, X):
        distances, indices = self.kneighbors(X, return_distance=True)
        return self._predict_precomputed(indices, distances)


class BatchedKNNClassifier(KNNClassifier):
    '''
    Нам нужен этот класс, потому что мы хотим поддержку обработки батчами
    в том числе для классов поиска соседей из sklearn
    '''

    def __init__(self, n_neighbors, algorithm='my_own', metric='euclidean', weights='uniform', batch_size=None):
        KNNClassifier.__init__(
            self,
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            weights=weights,
            metric=metric,
        )
        self._batch_size = batch_size

    def kneighbors(self, X, return_distance=False):
        if self._batch_size is None or self._batch_size >= X.shape[0]:
            return super().kneighbors(X, return_distance=return_distance)

        if not return_distance:
            # Bash course made me write oneliners
            return np.concatenate([
                super().kneighbors(X[lhs: lhs + self._batch_size], return_distance=return_distance)
                for lhs in range(0, len(X), self._batch_size)
            ])

        else:
            indices = []
            distances = []

            for lhs in range(0, len(X), self._batch_size):
                ind, dist = super().kneighbors(X[lhs: lhs + self._batch_size], return_distance=return_distance)
                
                indices.append(ind)
                distances.append(dist)

            return np.concatenate(indices), np.concatenate(distances)

 
