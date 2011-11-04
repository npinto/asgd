"""Averaging Stochastic Gradient Descent Classifier

naive, non-optimized implementation
"""

import numpy as np
from numpy import dot
from itertools import izip


class NaiveBinaryASGD(object):

    def __init__(self, n_features, sgd_step_size0=1e-2, l2_regularization=1e-3,
                 n_iterations=10, feedback=False, dtype=np.float32):

        self.n_features = n_features
        self.n_iterations = n_iterations
        self.feedback = feedback

        assert l2_regularization > 0
        self.l2_regularization = l2_regularization
        self.dtype = dtype

        self.sgd_weights = np.zeros((n_features), dtype=dtype)
        self.sgd_bias = np.zeros((1), dtype=dtype)
        self.sgd_step_size0 = sgd_step_size0
        self.sgd_step_size = sgd_step_size0
        self.sgd_step_size_scheduling_exponent = 2. / 3
        self.sgd_step_size_scheduling_multiplier = l2_regularization

        self.asgd_weights = np.zeros((n_features), dtype=dtype)
        self.asgd_bias = np.zeros((1), dtype=dtype)
        self.asgd_step_size0 = 1
        self.asgd_step_size = self.asgd_step_size0

        self.n_observations = 0

    def partial_fit(self, X, y):

        sgd_step_size0 = self.sgd_step_size0
        sgd_step_size = self.sgd_step_size
        sgd_step_size_scheduling_exponent = \
                self.sgd_step_size_scheduling_exponent
        sgd_step_size_scheduling_multiplier = \
                self.sgd_step_size_scheduling_multiplier
        sgd_weights = self.sgd_weights
        sgd_bias = self.sgd_bias

        asgd_weights = self.asgd_weights
        asgd_bias = self.asgd_bias
        asgd_step_size = self.asgd_step_size

        l2_regularization = self.l2_regularization

        n_observations = self.n_observations

        for obs, label in izip(X, y):

            # 1. compute margin
            margin = label * (dot(obs, sgd_weights) + sgd_bias)

            # 2.2 update sgd
            if l2_regularization:
                sgd_weights *= (1 - l2_regularization * sgd_step_size)

            if margin < 1:

                sgd_weights += sgd_step_size * label * obs
                sgd_bias += sgd_step_size * label

            # 2.2 update asgd
            asgd_weights = (1 - asgd_step_size) * asgd_weights \
                    + asgd_step_size * sgd_weights
            asgd_bias = (1 - asgd_step_size) * asgd_bias \
                    + asgd_step_size * sgd_bias

            # 4.1 update step_sizes
            n_observations += 1
            sgd_step_size_scheduling = (1 + sgd_step_size0 * n_observations *
                                        sgd_step_size_scheduling_multiplier)
            sgd_step_size = sgd_step_size0 / \
                    (sgd_step_size_scheduling ** \
                     sgd_step_size_scheduling_exponent)
            asgd_step_size = 1. / n_observations

        # --
        self.sgd_weights = sgd_weights
        self.sgd_bias = sgd_bias
        self.sgd_step_size = sgd_step_size

        self.asgd_weights = asgd_weights
        self.asgd_bias = asgd_bias
        self.asgd_step_size = asgd_step_size

        self.n_observations = n_observations

    def fit(self, X, y):

        assert X.ndim == 2
        assert y.ndim == 1

        n_points, n_features = X.shape
        assert n_features == self.n_features
        assert n_points == y.size

        n_iterations = self.n_iterations

        for i in xrange(n_iterations):

            idx = np.random.permutation(n_points)
            Xb = X[idx]
            yb = y[idx]
            self.partial_fit(Xb, yb)

            if self.feedback:
                self.sgd_weights = self.asgd_weights
                self.sgd_bias = self.asgd_bias

    def decision_function(self, X):
        return dot(self.asgd_weights, X.T) + self.asgd_bias

    def predict(self, X):
        return np.sign(self.decision_function(X))
