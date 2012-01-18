
import numpy as np
from itertools import izip

from naive_asgd import BaseASGD

from naive_asgd import (
        DEFAULT_SGD_STEP_SIZE0,
        DEFAULT_L2_REGULARIZATION,
        DEFAULT_N_ITERATIONS,
        DEFAULT_FEEDBACK,
        DEFAULT_RSTATE,
        DEFAULT_DTYPE,
        DEFAULT_SGD_EXPONENT,
        DEFAULT_SGD_TIMESCALE)

import theano
import theano.ifelse
import theano.tensor as tensor

class TheanoBinaryASGD(BaseASGD):

    """
    Notes regarding speed:
    1. This could be sped up futher by implementing an sdot Op in e.g.
       theano's blas_c.py file.  (currently _dot22 is used for inner product)

    2. Algorithmically, http://www.dbs.ifi.lmu.de/~yu_k/cvpr11_0694.pdf
    describes a change of variables that can potentially reduce the number of
    updates required by the algorithm, but I think it returns the same
    advantage as loop fusion / proper BLAS usage.
    """

    sgd_weights = property(lambda self: self.s_sgd_weights.get_value(),
            lambda self, val: self.s_sgd_weights.set_value(val))

    sgd_bias = property(lambda self: self.s_sgd_bias.get_value(),
            lambda self, val: self.s_sgd_weights.set_value(val))

    asgd_weights = property(lambda self: self.s_asgd_weights.get_value(),
            lambda self, val: self.s_asgd_weights.set_value(val))

    asgd_bias = property(lambda self: self.s_asgd_bias.get_value(),
            lambda self, val: self.s_asgd_bias.set_value(val))

    def __init__(self, n_features,
            sgd_step_size0=DEFAULT_SGD_STEP_SIZE0,
            l2_regularization=DEFAULT_L2_REGULARIZATION,
            n_iterations=DEFAULT_N_ITERATIONS,
            feedback=DEFAULT_FEEDBACK,
            rstate=DEFAULT_RSTATE,
            dtype=DEFAULT_DTYPE,
            sgd_step_size_scheduling_exponent=DEFAULT_SGD_EXPONENT,
            sgd_step_size_scheduling_multiplier=DEFAULT_SGD_TIMESCALE):

        BaseASGD.__init__(self,
            n_features,
            sgd_step_size0=sgd_step_size0,
            l2_regularization=l2_regularization,
            n_iterations=n_iterations,
            feedback=feedback,
            rstate=rstate,
            dtype=dtype,
            sgd_step_size_scheduling_exponent=sgd_step_size_scheduling_exponent,
            sgd_step_size_scheduling_multiplier=sgd_step_size_scheduling_multiplier)

        self.s_sgd_weights = theano.shared(
                np.zeros((n_features), dtype=dtype),
                name='sgd_weights')
        self.s_sgd_bias = theano.shared(
                np.asarray(0, dtype=dtype),
                name='sgd_bias')

        self.s_asgd_weights = theano.shared(
                np.zeros((n_features), dtype=dtype),
                name='asgd_weights')
        self.s_asgd_bias = theano.shared(
                np.asarray(0, dtype=dtype),
                name='asgd_bias')

        self.s_n_observations = theano.shared(np.asarray(0).astype('int64'))
        del self.n_observations

        self.s_sgd_step_size = theano.shared(np.asarray(0).astype(self.dtype))
        del self.sgd_step_size

    def vector_updates(self, obs, label):
        sgd_step_size0 = self.sgd_step_size0
        sgd_step_size_scheduling_exponent = \
                self.sgd_step_size_scheduling_exponent
        sgd_step_size_scheduling_multiplier = \
                self.sgd_step_size_scheduling_multiplier
        l2_regularization = self.l2_regularization

        n_observations = self.s_n_observations

        sgd_weights = self.s_sgd_weights
        sgd_bias = self.s_sgd_bias
        sgd_step_size = self.s_sgd_step_size

        asgd_weights = self.s_asgd_weights
        asgd_bias = self.s_asgd_bias
        asgd_step_size = 1.0 / (n_observations + 1)

        sgd_n = (1 + sgd_step_size0 * n_observations * sgd_step_size_scheduling_multiplier)
        sgd_step_size = (sgd_step_size0 /
                (sgd_n ** sgd_step_size_scheduling_exponent))

        # Use tensor.dot once the PR for the inner() optimization goes through
        # margin = label * (tensor.dot(obs, sgd_weights) + sgd_bias)
        margin = label * ((obs * sgd_weights).sum() + sgd_bias)
        regularized_sgd_weights = sgd_weights * tensor.cast(
                1 - l2_regularization * sgd_step_size,
                sgd_weights.dtype)

        if 1:
            switch = tensor.switch
        else:
            # this is theoretically better, but still slower
            # because of the linker's interaction with python to do lazy
            # evaluation
            switch = theano.ifelse.ifelse

        new_sgd_weights = switch(margin < 1,
                tensor.cast(
                    regularized_sgd_weights + sgd_step_size * label * obs,
                    regularized_sgd_weights.dtype),
                regularized_sgd_weights)
        new_sgd_bias = switch(margin < 1,
                tensor.cast(sgd_bias + sgd_step_size * label,
                    sgd_bias.dtype),
                sgd_bias)

        aa = tensor.cast(1 - asgd_step_size, asgd_weights.dtype)
        bb = tensor.cast(asgd_step_size, asgd_weights.dtype)

        new_asgd_weights = aa * asgd_weights + bb * new_sgd_weights
        new_asgd_bias = aa * asgd_bias + bb * new_sgd_bias

        new_n_observations = n_observations + 1

        updates = {
                    sgd_weights: new_sgd_weights,
                    sgd_bias: new_sgd_bias,
                    asgd_weights: new_asgd_weights,
                    asgd_bias: new_asgd_bias,
                    n_observations: new_n_observations}
        return updates

    def compile_train_fn_2(self):
        # This calling strategy is fast, but depends on the use of the CVM
        # linker.
        self._tf2_obs = obs = theano.shared(np.zeros((2, 2), dtype=self.dtype))
        self._tf2_label = label = theano.shared(np.zeros(2, dtype='int64'))
        self._tf2_idx = idx = theano.shared(np.asarray(0, dtype='int64'))
        self._tf2_idxmap = idxmap = theano.shared(np.zeros(2, dtype='int64'))
        updates = self.vector_updates(obs[idxmap[idx]], label[idxmap[idx]])
        updates[idx] = idx + 1
        self._train_fn_2 = theano.function([], [],
                updates=updates,
                mode=theano.Mode(
                    optimizer='fast_run',
                    linker='cvm'))

    def partial_fit(self, X, y):
        if '_train_fn_2' not in self.__dict__:
            self.compile_train_fn_2()
        n_points, n_features = X.shape
        assert n_features == self.n_features
        assert (n_points,) == y.shape

        self._tf2_obs.set_value(np.asarray(X, dtype=self.dtype), borrow=True)
        self._tf2_label.set_value(y, borrow=True)
        fn = self._train_fn_2.fn
        self._tf2_idxmap.set_value(np.arange(n_points))
        self._tf2_idx.set_value(0)
        if self._train_fn_2.profile:
            for i in xrange(n_points): self._train_fn_2()
        else:
            for i in xrange(n_points): fn()

    def fit(self, X, y):
        if '_train_fn_2' not in self.__dict__:
            self.compile_train_fn_2()

        assert X.ndim == 2
        assert y.ndim == 1

        n_points, n_features = X.shape
        assert n_features == self.n_features
        assert n_points == y.size

        n_iterations = self.n_iterations

        self._tf2_obs.set_value(X, borrow=True)
        self._tf2_label.set_value(y, borrow=True)
        fn = self._train_fn_2.fn

        for i in xrange(n_iterations):
            self._tf2_idxmap.set_value(self.rstate.permutation(n_points))
            self._tf2_idx.set_value(0)
            if self._train_fn_2.profile:
                for i in xrange(n_points): self._train_fn_2()
            else:
                for i in xrange(n_points): fn()

    def decision_function(self, X):
        return (np.dot(self.s_asgd_weights.get_value(borrow=True), X.T)
                + self.asgd_bias)

    def predict(self, X):
        return np.sign(self.decision_function(X))
