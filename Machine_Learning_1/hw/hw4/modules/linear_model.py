import numpy as np
from scipy.special import expit
import time
from collections import defaultdict

class LinearModel:
    def __init__(
        self,
        loss_function,
        batch_size=None,
        step_alpha=1,
        step_beta=0, 
        tolerance=1e-5,
        max_iter=1000,
        random_seed=911,
        **kwargs
    ):
        """
        Parameters
        ----------
        loss_function : BaseLoss inherited instance
            Loss function to use
        batch_size : int
        step_alpha : float
        step_beta : float
            step_alpha and step_beta define the learning rate behaviour
        tolerance : float
            Tolerace for stop criterio.
        max_iter : int
            Max amount of epoches in method.
        """
        self.loss_function = loss_function
        self.batch_size = batch_size
        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        np.random.seed(random_seed)
        self.random_seed = random_seed

    def fit(self, X, y, w_0=None, trace=False, X_val=None, y_val=None):
        """

        Parameters
        ----------
        X : numpy.ndarray
            2d matrix, training set.
        y : numpy.ndarray
            1d vector, target values.
        w_0 : numpy.ndarray
            1d vector in binary classification.
            Initial approximation for SGD method - [bias, weights]
        trace : bool
            If True need to calculate metrics on each iteration.
        X_val : numpy.ndarray
            2d matrix, validation set.
        y_val: numpy.ndarray
            1d vector, target values for validation set.

        Returns
        -------
        : dict
            Keys are 'time', 'func', 'func_val'.
            Each key correspond to list of metric values after each training epoch.
        """

        permutation = np.random.permutation(len(X))
        X = X[permutation]
        y = y[permutation]
        
        bath_size = self.batch_size if self.batch_size is not None else len(X)
        self.w = w_0.copy() if w_0 is not None else np.ones(X.shape[1] + 1)
        history = defaultdict(list)
        prev_loss = None

        for k in range(self.max_iter): # Epoch iter
            time_start = time.time()

            step = self.step_alpha / (k + 1) ** self.step_beta

            for bsi in range(0, len(X), bath_size): # batch start index
                grad = self.loss_function.grad(
                    X[bsi: bsi + bath_size], 
                    y[bsi: bsi + bath_size], 
                    self.w
                )
                self.w -= step * grad

            cur_loss = self.loss_function.func(X, y, self.w)
            if prev_loss is not None and np.abs(cur_loss - prev_loss) < self.tolerance:
                break

            prev_loss = cur_loss

            time_end = time.time()

            if trace:
                history['time'].append(time_end - time_start)
                history['func'].append(cur_loss)
                history['func_val'].append(self.loss_function.func(X_val, y_val, self.w))
        
        if trace:
            return history
                
    def predict(self, X, threshold=0):
        """

        Parameters
        ----------
        X : numpy.ndarray
            2d matrix, test set.
        threshold : float
            Chosen target binarization threshold.

        Returns
        -------
        : numpy.ndarray
            answers on a test set
        """
        

        return (
            (expit(np.hstack((np.ones((len(X), 1)), X)) @ self.w) * 2 - 1) > threshold
        ).astype(int) * 2 - 1

    def get_optimal_threshold(self, X, y):
        """
        Get optimal target binarization threshold.
        Balanced accuracy metric is used.

        Parameters
        ----------
        X : numpy.ndarray
            2d matrix, validation set.
        y : numpy.ndarray
            1d vector, target values for validation set.

        Returns
        -------
        : float
            Chosen threshold.
        """

        weights, bias = self.w[1:], self.w[0]
        scores = X.dot(weights) + bias
        y_to_index = {-1: 0, 1: 1}

        # for each score store real targets that correspond score
        score_to_y = dict()
        score_to_y[min(scores) - 1e-5] = [0, 0]
        for one_score, one_y in zip(scores, y):
            score_to_y.setdefault(one_score, [0, 0])
            score_to_y[one_score][y_to_index[one_y]] += 1

        # ith element of cum_sums is amount of y <= alpha
        scores, y_counts = zip(*sorted(score_to_y.items(), key=lambda x: x[0]))
        cum_sums = np.array(y_counts).cumsum(axis=0)

        # count balanced accuracy for each threshold
        recall_for_negative = cum_sums[:, 0] / cum_sums[-1][0]
        recall_for_positive = 1 - cum_sums[:, 1] / cum_sums[-1][1]
        ba_accuracy_values = 0.5 * (recall_for_positive + recall_for_negative)
        best_score = scores[np.argmax(ba_accuracy_values)]
        return best_score

    def get_weights(self):
        """
        Get model weights (w[1:])

        Returns
        -------
        : numpy.ndarray
            1d vector in binary classification.
            Initial approximation for SGD method.
        """
        return self.w[1:]

    def get_bias(self):
        """
        Get model bias

        Returns
        -------
        : float
        """
        return self.w[0]

    def get_objective(self, X, y):
        """
        Get objective function value.

        Parameters
        ----------
        X : numpy.ndarray
            2d matrix.
        y : numpy.ndarray
            1d vector, target values for X.

        Returns
        -------
        : float
        """
        
        return self.loss_function.func(np.hstack((np.ones(len(X)), X)), y, self.w)