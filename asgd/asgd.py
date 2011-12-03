"""Averaging Stochastic Gradient Descent Classifier

naive, non-optimized implementation
"""
import ctypes as ct
import numpy as np
from numpy import dot
from itertools import izip


class ASGD(object):

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
        self.core_lib = ct.CDLL("./asgd_core.so")
        

    def partial_fit(self, X, y):

        input_req = ['A', 'O', 'W', 'C']
        output_req = ['A', 'O', 'W', 'C']
        np.require(self.sgd_weights, dtype=np.float32, requirements=output_req)
        np.require(self.sgd_bias, dtype=np.float32, requirements=output_req)
        np.require(self.asgd_weights, dtype=np.float32, requirements=output_req)
        np.require(self.asgd_bias, dtype=np.float32, requirements=output_req)
        np.require(X, dtype=np.float32, requirements=input_req)
        np.require(y, dtype=np.float32, requirements=input_req)
        
        sgd_step_size0 = ct.c_float(self.sgd_step_size0)
        sgd_step_size = ct.c_float(self.sgd_step_size)
        sgd_step_size_scheduling_exponent = \
                ct.c_float(self.sgd_step_size_scheduling_exponent)
        sgd_step_size_scheduling_multiplier = \
                ct.c_float(self.sgd_step_size_scheduling_multiplier)
        sgd_weights = self.sgd_weights
        sgd_bias = self.sgd_bias

        asgd_weights = self.asgd_weights
        asgd_bias = self.asgd_bias
        asgd_step_size = ct.c_float(self.asgd_step_size)

        l2_regularization = ct.c_float(self.l2_regularization)

        n_observations = ct.c_long(self.n_observations)

        self.core_lib.core_partial_fit(
                ct.byref(n_observations),\
                ct.byref(sgd_step_size),\
                ct.byref(asgd_step_size),\
                l2_regularization,\
                sgd_step_size0,\
                sgd_step_size_scheduling_exponent,\
                sgd_step_size_scheduling_multiplier,\
                sgd_weights.ctypes.data_as(ct.POINTER(ct.c_float)),\
                ct.c_size_t(sgd_weights.shape[0]),\
                ct.c_size_t(1),\
                sgd_bias.ctypes.data_as(ct.POINTER(ct.c_float)),\
                ct.c_size_t(1),\
                ct.c_size_t(1),\
                asgd_weights.ctypes.data_as(ct.POINTER(ct.c_float)),\
                ct.c_size_t(asgd_weights.shape[0]),\
                ct.c_size_t(1),\
                asgd_bias.ctypes.data_as(ct.POINTER(ct.c_float)),\
                ct.c_size_t(1),\
                ct.c_size_t(1),\
                X.ctypes.data_as(ct.POINTER(ct.c_float)),\
                ct.c_size_t(X.shape[0]),\
                ct.c_size_t(X.shape[1]),\
                y.ctypes.data_as(ct.POINTER(ct.c_float)),\
                ct.c_size_t(y.shape[0]),\
                ct.c_size_t(y.shape[1]))

        # --
        self.sgd_weights = sgd_weights
        self.sgd_bias = sgd_bias
        self.sgd_step_size = sgd_step_size.value

        self.asgd_weights = asgd_weights
        self.asgd_bias = asgd_bias
        self.asgd_step_size = asgd_step_size.value

        self.n_observations = n_observations.value

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
