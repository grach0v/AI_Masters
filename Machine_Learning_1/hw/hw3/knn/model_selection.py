from collections import defaultdict

import numpy as np

from sklearn.model_selection import KFold, BaseCrossValidator
from sklearn.metrics import accuracy_score

from knn.classification import BatchedKNNClassifier
from collections import defaultdict


def knn_cross_val_score(X, y, k_list, scoring, cv=None, **kwargs):
    y = np.asarray(y)

    if scoring == "accuracy":
        scorer = accuracy_score
    else:
        raise ValueError("Unknown scoring metric", scoring)

    if cv is None:
        cv = KFold(n_splits=5)
    elif not isinstance(cv, BaseCrossValidator):
        raise TypeError("cv should be BaseCrossValidator instance", type(cv))

    ans = defaultdict(list)

    for train_index, test_index in cv.split(X):
        model = BatchedKNNClassifier(n_neighbors=max(k_list), **kwargs)
        model.fit(X[train_index], y[train_index])
        distances, indices = model.kneighbors(X[test_index], return_distance=True)

        for k in k_list:
            # Is this ok?
            model._finder.n_neighbors = k
            y_pred = model._predict_precomputed(indices[:, :k], distances[:, :k])
            ans[k].append(scorer(y[test_index], y_pred))

    for k in k_list:
        ans[k] = np.array(ans[k])

    return ans

    # for k in k_list:
    #     ans[k] = []

    #     for train_index, test_index in cv.split(X):
    #         model = BatchedKNNClassifier(n_neighbors=k)
    #         model.fit(X[train_index], y[train_index])
    #         y_pred = model.predict(X[test_index])

    #         ans[k].append(scorer(y[test_index], y_pred))
        
    #     ans[k] = np.array(ans[k])
    
    # return ans

    # Now this is a challenge
    # I will try to make it readable
    # return {
    #     k: np.array([
    #         scorer(
    #             BatchedKNNClassifier(n_neighbors=k, **kwargs).fit(X[train_index], y[train_index]).predict(X[test_index]),
    #             y[test_index],
    #         )
    #         for train_index, test_index in cv.split(X)
    #     ])
    #     for k in k_list
    # }

