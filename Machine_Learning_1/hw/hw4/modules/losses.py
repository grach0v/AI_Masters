import numpy as np
import scipy
from scipy.special import expit
from scipy.special import logsumexp


class BaseLoss:
    """
    Base class for loss function.
    """

    def func(self, X, y, w):
        """
        Get loss function value at w.
        """
    
        return np.mean(np.logaddexp(0, -y * (X @ w)))

    def grad(self, X, y, w):
        """
        Get loss function gradient value at w.
        """

        grad = np.mean(
            -y.reshape(-1, 1) * X * (
                1 - expit(y * (X @ w)).reshape(-1, 1)
            ),
            axis=0
        )
        return grad

class BinaryLogisticLoss(BaseLoss):
    """
    Loss function for binary logistic regression.
    It should support l2 regularization.
    """

    def __init__(self, l2_coef):
        """
        Parameters
        ----------
        l2_coef - l2 regularization coefficient
        """
        self.l2_coef = l2_coef

    def func(self, X, y, w):
        """
        Get loss function value for data X, target y and coefficient w; w = [bias, weights].

        Parameters
        ----------
        X : 2d numpy.ndarray
        y : 1d numpy.ndarray
        w : 1d numpy.ndarray

        Returns
        -------
        : float
        """

        return super().func(
            np.hstack([np.ones((len(X), 1)), X]), 
            y, 
            w
        ) + self.l2_coef * (w[1:] @ w[1:])

    def grad(self, X, y, w):
        """
        Get loss function gradient for data X, target y and coefficient w; w = [bias, weights].

        Parameters
        ----------
        X : 2d numpy.ndarray
        y : 1d numpy.ndarray
        w : 1d numpy.ndarray

        Returns
        -------
        : 1d numpy.ndarray
        """
        
        return super().grad(
            np.hstack([np.ones((len(X), 1)), X]), 
            y, 
            w
        ) + np.asarray([0, *(2 * self.l2_coef * w[1:])])


