
import numpy as np
from itertools import izip

from naive_asgd import BaseASGD
from naive_asgd import DetermineStepSizeMixin

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


class TheanoBinaryASGD(BaseASGD, DetermineStepSizeMixin):

    """
    Notes regarding speed:
    1. This could be sped up futher by implementing an sdot Op in e.g.
       theano's blas_c.py file.  (currently _dot22 is used for inner product)

    2. Algorithmically, http://www.dbs.ifi.lmu.de/~yu_k/cvpr11_0694.pdf
    describes a change of variables that can potentially reduce the number of
    updates required by the algorithm, but I think it returns the same
    advantage as loop fusion / proper BLAS usage.
    """

    sgd_weights = property(
            lambda self: self.s_sgd_weights.get_value(),
            lambda self, val: self.s_sgd_weights.set_value(val))

    sgd_bias = property(
            lambda self: self.s_sgd_bias.get_value(),
            lambda self, val: self.s_sgd_bias.set_value(np.asarray(val)))

    asgd_weights = property(
            lambda self: self.s_asgd_weights.get_value(),
            lambda self, val: self.s_asgd_weights.set_value(val))

    asgd_bias = property(
            lambda self: self.s_asgd_bias.get_value(),
            lambda self, val: self.s_asgd_bias.set_value(np.asarray(val)))

    n_observations = property(
            lambda self: self.s_n_observations.get_value(),
            lambda self, val: self.s_n_observations.set_value(np.asarray(val)))

    sgd_step_size0 = property(
            lambda self: self.s_sgd_step_size0.get_value(),
            lambda self, val: self.s_sgd_step_size0.set_value(np.asarray(val)))

    use_switch = False

    def __init__(self, n_features,
            sgd_step_size0=DEFAULT_SGD_STEP_SIZE0,
            l2_regularization=DEFAULT_L2_REGULARIZATION,
            n_iterations=DEFAULT_N_ITERATIONS,
            feedback=DEFAULT_FEEDBACK,
            rstate=DEFAULT_RSTATE,
            dtype=DEFAULT_DTYPE,
            sgd_step_size_scheduling_exponent=DEFAULT_SGD_EXPONENT,
            sgd_step_size_scheduling_multiplier=DEFAULT_SGD_TIMESCALE):

        self.s_n_observations = theano.shared(
                np.asarray(0).astype('int64'),
                name='n_observations',
                allow_downcast=True)

        self.s_sgd_step_size0 = theano.shared(
                np.asarray(0).astype(dtype),
                name='sgd_step_size0',
                allow_downcast=True)

        self.s_sgd_weights = theano.shared(
                np.zeros((n_features), dtype=dtype),
                name='sgd_weights',
                allow_downcast=True)

        self.s_sgd_bias = theano.shared(
                np.asarray(0, dtype=dtype),
                name='sgd_bias')

        self.s_asgd_weights = theano.shared(
                np.zeros((n_features), dtype=dtype),
                name='asgd_weights')

        self.s_asgd_bias = theano.shared(
                np.asarray(0, dtype=dtype),
                name='asgd_bias')

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

    def __getstate__(self):
        dct = dict(self.__dict__)
        dynamic_attrs = [
                '_train_fn_2',
                '_tf2_obs',
                '_tf2_idx',
                '_tf2_idxmap',
                '_tf2_mean_cost']
        for attr in dynamic_attrs:
            if attr in dct:
                del dct[attr]
        return dct

    def vector_updates(self, obs, label):
        sgd_step_size_scheduling_exponent = \
                self.sgd_step_size_scheduling_exponent
        sgd_step_size_scheduling_multiplier = \
                self.sgd_step_size_scheduling_multiplier
        l2_regularization = self.l2_regularization

        n_observations = self.s_n_observations
        sgd_step_size0 = self.s_sgd_step_size0

        sgd_weights = self.s_sgd_weights
        sgd_bias = self.s_sgd_bias
        sgd_n = (1 +
                sgd_step_size0 * n_observations
                * sgd_step_size_scheduling_multiplier)
        sgd_step_size = tensor.cast(
                (sgd_step_size0
                    / (sgd_n ** sgd_step_size_scheduling_exponent)),
                sgd_weights.dtype)

        asgd_weights = self.s_asgd_weights
        asgd_bias = self.s_asgd_bias

        # switch to geometric moving average after a while
        # ALSO - this means that mul rather than div is used in the fused
        # elementwise loop that updates asgd_weights, which is faster
        asgd_step_size = tensor.maximum(
                tensor.cast(
                    1.0 / (n_observations + 1),
                    asgd_bias.dtype),
                1e-5)


        margin = label * (tensor.dot(obs, sgd_weights) + sgd_bias)
        regularized_sgd_weights = sgd_weights * tensor.cast(
                1 - l2_regularization * sgd_step_size,
                sgd_weights.dtype)

        if self.use_switch:
            switch = tensor.switch
        else:
            # this is slower to evaluate, but if the features are long and the
            # expected training classification rate is very good, then it
            # can be faster.
            switch = theano.ifelse.ifelse

        assert regularized_sgd_weights.dtype == sgd_weights.dtype
        assert obs.dtype == sgd_weights.dtype
        assert label.dtype == sgd_weights.dtype
        assert sgd_step_size.dtype == sgd_weights.dtype

        new_sgd_weights = switch(margin < 1,
                regularized_sgd_weights + sgd_step_size * label * obs,
                regularized_sgd_weights)
        new_sgd_bias = switch(margin < 1,
                sgd_bias + sgd_step_size * label,
                1 * sgd_bias)
        cost = switch(margin < 1, 1 - margin, 0 * margin)

        new_asgd_weights = ((1.0 - asgd_step_size) * asgd_weights
            + asgd_step_size * new_sgd_weights)
        new_asgd_bias = ((1.0 - asgd_step_size) * asgd_bias
                + asgd_step_size * new_sgd_bias)

        new_n_observations = n_observations + 1

        updates = {
                    sgd_weights: new_sgd_weights,
                    sgd_bias: new_sgd_bias,
                    asgd_weights: new_asgd_weights,
                    asgd_bias: new_asgd_bias,
                    n_observations: new_n_observations}
        return updates, cost

    def compile_train_fn_2(self):
        # This calling strategy is fast, but depends on the use of the CVM
        # linker.
        self._tf2_obs = obs = theano.shared(np.zeros((2, 2),
            dtype=self.dtype),
            allow_downcast=True,
            name='obs')
        # N.B. labels are float
        self._tf2_label = label = theano.shared(np.zeros(2, dtype=self.dtype),
                allow_downcast=True,
                name='label')
        self._tf2_idx = idx = theano.shared(np.asarray(0, dtype='int64'))
        self._tf2_idxmap = idxmap = theano.shared(np.zeros(2, dtype='int64'),
                strict=True)
        self._tf2_mean_cost = mean_cost = theano.shared(
                np.asarray(0, dtype='float64'))
        updates, cost = self.vector_updates(obs[idxmap[idx]], label[idxmap[idx]])
        updates[idx] = idx + 1
        aa = tensor.cast(1.0 / (idx + 1), 'float64')
        # mean cost over idxmap
        updates[mean_cost] = (1 - aa) * mean_cost + aa * cost

        self._train_fn_2 = theano.function([], [],
                updates=updates,
                mode=theano.Mode(
                    optimizer='fast_run',
                    linker='cvm_nogc'))

    def partial_fit(self, X, y):
        if '_train_fn_2' not in self.__dict__:
            self.compile_train_fn_2()
        n_points, n_features = X.shape
        assert n_features == self.n_features
        assert (n_points,) == y.shape
        assert np.all(y ** 2 == 1)  # make sure labels are +-1

        self._tf2_obs.set_value(X, borrow=True)
        # This may cast `y` to a floating point type
        self._tf2_label.set_value(y, borrow=True)
        self._tf2_idxmap.set_value(np.arange(n_points), borrow=True)
        self._tf2_idx.set_value(0)

        if self._train_fn_2.profile:
            for i in xrange(n_points): self._train_fn_2()
        else:
            fn = self._train_fn_2.fn
            for i in xrange(n_points): fn()

        self.train_means.append(self._tf2_mean_cost.get_value()
                    + self.l2_regularization * (self.asgd_weights ** 2).sum())
        return self

    def fit(self, X, y):
        if '_train_fn_2' not in self.__dict__:
            self.compile_train_fn_2()

        if self.sgd_step_size0 is None:
            self.determine_sgd_step_size0(X, y)

        assert X.ndim == 2
        assert y.ndim == 1
        assert np.all(y ** 2 == 1)  # make sure labels are +-1

        n_points, n_features = X.shape
        assert n_features == self.n_features
        assert n_points == y.size

        n_iterations = self.n_iterations
        train_means = self.train_means

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
            train_means.append(self._tf2_mean_cost.get_value()
                    + self.l2_regularization * (self.asgd_weights ** 2).sum())
            if self.fit_converged():
                break

        return self

    def decision_function(self, X):
        return (np.dot(self.s_asgd_weights.get_value(borrow=True), X.T)
                + self.asgd_bias)

    def predict(self, X):
        return np.sign(self.decision_function(X))

    def reset(self):
        BaseASGD.reset(self)
        self.asgd_weights = self.asgd_weights * 0
        self.asgd_bias = self.asgd_bias * 0
        self.sgd_weights = self.sgd_weights * 0
        self.sgd_bias = self.sgd_bias * 0
