"""Averaging Stochastic Gradient Descent Classifier

naive, non-optimized implementation
"""

import copy
from itertools import izip

import numpy as np
from numpy import dot

from .utils import geometric_bracket_min

DEFAULT_SGD_STEP_SIZE0 = None
DEFAULT_L2_REGULARIZATION = 1e-3
DEFAULT_N_ITERATIONS = 10
DEFAULT_FEEDBACK = False
DEFAULT_RSTATE = 42
DEFAULT_DTYPE = np.float64
DEFAULT_SGD_EXPONENT = 2.0 / 3.0
DEFAULT_SGD_TIMESCALE = 'l2_regularization'
# can be 'l2_regularization' or float
# This timescale default comes from [1] in which it is introduced as a
# heuristic.
# [1] http://www.dbs.ifi.lmu.de/~yu_k/cvpr11_0694.pdf
# Update: it is also recommended in Leon Bottou's SvmAsgd software.


class BaseASGD(object):
    """
    XXX
    """

    min_n_iterations = 5

    def __init__(self, n_features,
                 sgd_step_size0=DEFAULT_SGD_STEP_SIZE0,
                 l2_regularization=DEFAULT_L2_REGULARIZATION,
                 n_iterations=DEFAULT_N_ITERATIONS,
                 feedback=DEFAULT_FEEDBACK,
                 rstate=DEFAULT_RSTATE,
                 dtype=DEFAULT_DTYPE,
                 sgd_step_size_scheduling_exponent = DEFAULT_SGD_EXPONENT,
                 sgd_step_size_scheduling_multiplier = DEFAULT_SGD_TIMESCALE):

        # --
        assert n_features > 1
        self.n_features = n_features

        assert n_iterations > 0
        self.n_iterations = n_iterations

        self.min_n_iterations = min(n_iterations, self.min_n_iterations)

        if feedback:
            raise NotImplementedError("FIXME: feedback support is buggy")
        self.feedback = feedback

        if rstate is None:
            rstate = np.random.RandomState()
        elif type(rstate) is int:
            rstate = np.random.RandomState(rstate)
        self.rstate = rstate

        self.l2_regularization = l2_regularization
        self.dtype = dtype

        # --
        self.sgd_step_size0 = sgd_step_size0
        self.sgd_step_size_scheduling_exponent = \
            sgd_step_size_scheduling_exponent
        if sgd_step_size_scheduling_multiplier == 'l2_regularization':
            self.sgd_step_size_scheduling_multiplier = l2_regularization
        else:
            self.sgd_step_size_scheduling_multiplier = \
                    sgd_step_size_scheduling_multiplier

        self.n_observations = 0
        self.asgd_step_size0 = 1
        self.asgd_step_size = self.asgd_step_size0
        self.sgd_step_size = self.sgd_step_size0
        self.train_means = []

    def fit_converged(self):
        train_means = self.train_means
        if len(train_means) >= self.min_n_iterations:
            midpt = len(train_means) // 2
            if train_means[-1] > .99 * train_means[midpt]:
                return True
        return False

    def test_error_rate(self, X, y):
        yhat = self.predict(X)
        mu = np.mean((y != yhat).astype('float'))
        var = mu * (1 - mu) / len(y)
        stderr = np.sqrt(var)
        return mu, stderr


class DetermineStepSizeMixin(object):
    """
    Implements the automatic step-size selection logic from
    http://leon.bottou.org/projects/sgd


    This mixin requires the host class to have

    self.partial_fit(X, y)
    self.sgd_step_size0
    self.sgd_step_size
    self.sgd_weights
    self.sgd_bias
    self.asgd_weights
    self.asgd_bias

    """

    n_examples_for_determining_step_size = 1000
    verbose = 0

    def determine_sgd_step_size0(self, X, y, base=1.0e-2, factor=2.0):
        # trim X and y down to at most 1000 examples
        X = X[:self.n_examples_for_determining_step_size]
        y = y[:self.n_examples_for_determining_step_size]
        def eval_size0(size0):
            rval = self.evaluate_step_size(X, y, size0)
            if self.verbose:
                print('determine_sgd_step_size0: %.6f\t%.6f' % (
                    size0, rval))
            return rval
        lo_size0, hi_size0 = geometric_bracket_min(eval_size0,
                x0=base,
                x1=factor * base,
                f_thresh=1e-4)
        # assume that the actual learning rate should be just a little slower
        # than was optimal on the first 1000 points
        self.sgd_step_size0 = lo_size0 / factor
        self.sgd_step_size = self.sgd_step_size0

    def evaluate_step_size(self, X, y, sgd_step_size0):
        other = copy.deepcopy(self)
        other.sgd_step_size0 = sgd_step_size0
        other.sgd_step_size = sgd_step_size0
        other.partial_fit(X, y)
        # XXX: hack - asgd is lower variance than sgd, but it's tuned to work
        #             well asymptotically, not after just a few examples
        weights = .5 * other.asgd_weights + .5 * other.sgd_weights
        bias = .5 * other.asgd_bias + .5 * other.sgd_bias
        margin = y * (dot(X, weights) + bias)
        l2_cost = other.l2_regularization * (weights ** 2).sum()
        cost = np.maximum(0, 1 - margin) + l2_cost
        return cost.mean()


class NaiveBinaryASGD(BaseASGD, DetermineStepSizeMixin):
    """
    XXX
    """

    def __init__(self, n_features, sgd_step_size0=DEFAULT_SGD_STEP_SIZE0,
                 l2_regularization=DEFAULT_L2_REGULARIZATION,
                 n_iterations=DEFAULT_N_ITERATIONS, feedback=DEFAULT_FEEDBACK,
                 rstate=DEFAULT_RSTATE, dtype=DEFAULT_DTYPE):

        super(NaiveBinaryASGD, self).__init__(
            n_features,
            sgd_step_size0=sgd_step_size0,
            l2_regularization=l2_regularization,
            n_iterations=n_iterations,
            feedback=feedback,
            rstate=rstate,
            dtype=dtype,
            )

        # --
        self.sgd_weights = np.zeros((n_features), dtype=dtype)
        self.sgd_bias = np.zeros((1), dtype=dtype)

        self.asgd_weights = np.zeros((n_features), dtype=dtype)
        self.asgd_bias = np.zeros((1), dtype=dtype)

    def partial_fit(self, X, y):

        assert np.all(y ** 2 == 1)  # make sure labels are +-1
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

        costs = []

        for obs, label in izip(X, y):

            # -- compute margin
            margin = label * (dot(obs, sgd_weights) + sgd_bias)

            # -- update sgd
            if l2_regularization:
                sgd_weights *= (1 - l2_regularization * sgd_step_size)

            if margin < 1:

                sgd_weights += sgd_step_size * label * obs
                sgd_bias += sgd_step_size * label
                costs.append(1 - float(margin))
            else:
                costs.append(0)

            # -- update asgd
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

        self.train_means.append(np.mean(costs)
                + self.l2_regularization * (self.asgd_weights ** 2).sum())

        return self

    def fit(self, X, y):

        assert X.ndim == 2
        assert y.ndim == 1

        n_points, n_features = X.shape
        assert n_features == self.n_features
        assert n_points == y.size

        n_iterations = self.n_iterations

        if self.sgd_step_size0 is None:
            self.determine_sgd_step_size0(X, y)

        for i in xrange(n_iterations):

            idx = self.rstate.permutation(n_points)
            Xb = X[idx]
            yb = y[idx]
            self.partial_fit(Xb, yb)

            if self.feedback:
                self.sgd_weights = self.asgd_weights
                self.sgd_bias = self.asgd_bias

            if self.fit_converged():
                break

        return self

    def decision_function(self, X):
        return dot(self.asgd_weights, X.T) + self.asgd_bias

    def predict(self, X):
        return np.sign(self.decision_function(X))

    def reset(self):
        BaseASGD.reset(self)
        self.asgd_weights = self.asgd_weights * 0
        self.asgd_bias = self.asgd_bias * 0
        self.sgd_weights = self.sgd_weights * 0
        self.sgd_bias = self.sgd_bias * 0


class NaiveOVAASGD(BaseASGD):
    """
    XXX
    """

    def __init__(self, n_classes, n_features,
                 sgd_step_size0=DEFAULT_SGD_STEP_SIZE0,
                 l2_regularization=DEFAULT_L2_REGULARIZATION,
                 n_iterations=DEFAULT_N_ITERATIONS,
                 feedback=DEFAULT_FEEDBACK,
                 rstate=DEFAULT_RSTATE,
                 dtype=DEFAULT_DTYPE):

        super(NaiveOVAASGD, self).__init__(
            n_features,
            sgd_step_size0=sgd_step_size0,
            l2_regularization=l2_regularization,
            n_iterations=n_iterations,
            feedback=feedback,
            rstate=rstate,
            dtype=dtype,
            )

        # --
        assert n_classes > 1
        self.n_classes = n_classes

        # --
        self.sgd_weights = np.zeros((n_features, n_classes), dtype=dtype)
        self.sgd_bias = np.zeros((n_classes,), dtype=dtype)
        self.asgd_weights = np.zeros((n_features, n_classes), dtype=dtype)
        self.asgd_bias = np.zeros((n_classes), dtype=dtype)

    def partial_fit(self, X, y):

        if set(y) > set(range(self.n_classes)):
            raise ValueError("Invalid 'y'")

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
        n_classes = self.n_classes

        for obs, label in izip(X, y):
            label = 2 * (np.arange(n_classes) == label).astype(int) - 1

            # -- compute margin
            margin = label * (dot(obs, sgd_weights) + sgd_bias)

            # -- update sgd
            if l2_regularization:
                sgd_weights *= (1 - l2_regularization * sgd_step_size)

            violations = margin < 1
            label_violated = label[violations]
            sgd_weights[:, violations] += (
                sgd_step_size
                * label_violated[np.newaxis, :]
                * obs[:, np.newaxis]
            )
            sgd_bias[violations] += sgd_step_size * label_violated

            # -- update asgd
            asgd_weights = (1 - asgd_step_size) * asgd_weights \
                    + asgd_step_size * sgd_weights
            asgd_bias = (1 - asgd_step_size) * asgd_bias \
                    + asgd_step_size * sgd_bias

            # -- update step_sizes
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

        return self

    def fit(self, X, y):

        assert X.ndim == 2
        assert y.ndim == 1

        n_points, n_features = X.shape
        assert n_features == self.n_features
        assert n_points == y.size

        n_iterations = self.n_iterations

        for i in xrange(n_iterations):

            idx = self.rstate.permutation(n_points)
            Xb = X[idx]
            yb = y[idx]
            self.partial_fit(Xb, yb)

            if self.feedback:
                self.sgd_weights = self.asgd_weights
                self.sgd_bias = self.asgd_bias

        return self

    def decision_function(self, X):
        return dot(X, self.asgd_weights) + self.asgd_bias

    def predict(self, X):
        return self.decision_function(X).argmax(1)
