import sys
import time
from copy import copy

from nose.tools import assert_equal, raises
from nose.plugins.skip import SkipTest
from numpy.testing import assert_allclose
import numpy as np
from numpy.random import RandomState

from asgd.naive_asgd import NaiveBinaryASGD
from test_naive_asgd import get_fake_data
try:
    from asgd.theano_asgd import TheanoBinaryASGD
except ImportError:
    print >> sys.stderr, "\nWARNING: SKIPPING ALL TESTS REQUIRING THEANO"

RTOL = 1e-6
ATOL = 1e-6

N_POINTS = 1e3
N_FEATURES = 1e2

DEFAULT_ARGS = (N_FEATURES,)
DEFAULT_KWARGS = dict(sgd_step_size0=1e-3,
                      l2_regularization=1e-6,
                      n_iterations=4,
                      dtype=np.float32)


def requires_theano(f):
    def rval():
        if 'TheanoBinaryASGD' in globals():
            f()
        else:
            raise SkipTest('TheanoBinaryASGD failed to import')
    rval.__name__ = f.__name__
    return rval


@requires_theano
def test_theano_binary_asgd_like_naive_asgd():

    rstate = RandomState(42)

    X, y = get_fake_data(N_POINTS, N_FEATURES, rstate)
    Xtst, ytst = get_fake_data(N_POINTS, N_FEATURES, rstate)

    clf0 = NaiveBinaryASGD(*DEFAULT_ARGS, rstate=copy(rstate), **DEFAULT_KWARGS)
    clf1 = TheanoBinaryASGD(*DEFAULT_ARGS, rstate=copy(rstate), **DEFAULT_KWARGS)
    clf1.min_n_iterations = clf1.n_iterations

    for clf in [clf0, clf1]:
        clf.fit(X, y)
        ytrn_preds = clf.predict(X)
        ytst_preds = clf.predict(Xtst)
        ytrn_acc = (ytrn_preds == y).mean()
        ytst_acc = (ytst_preds == y).mean()
        assert_equal(ytrn_acc, 0.72)
        assert_equal(ytst_acc, 0.522)


@requires_theano
def test_theano_binary_asgd_early_stopping():

    rstate = RandomState(42)

    X, y = get_fake_data(N_POINTS, N_FEATURES, rstate)
    Xtst, ytst = get_fake_data(N_POINTS, N_FEATURES, rstate)

    kwargs = dict(DEFAULT_KWARGS)
    kwargs['n_iterations'] = 30

    clf0 = NaiveBinaryASGD(*DEFAULT_ARGS, rstate=copy(rstate), **kwargs)
    clf1 = TheanoBinaryASGD(*DEFAULT_ARGS, rstate=copy(rstate), **kwargs)

    clf0.fit(X, y)
    clf1.fit(X, y)

    assert clf0.n_observations == clf1.n_observations
    assert clf0.n_observations < N_POINTS * kwargs['n_iterations']



@requires_theano
def test_theano_binary_asgd_converges_to_truth():
    n_features = 5

    rstate = RandomState(42)

    true_weights = rstate.randn(n_features)
    true_bias = rstate.randn() * 0
    def draw_data(N=20):
        X = rstate.randn(N, 5)
        labels = np.sign(np.dot(X, true_weights) + true_bias).astype('int')
        return X, labels

    clf = TheanoBinaryASGD(n_features,
            rstate=rstate,
            sgd_step_size0=0.1,
            dtype='float64',
            l2_regularization = 1e-4,
            sgd_step_size_scheduling_exponent=0.5,
            sgd_step_size_scheduling_multiplier=1.0)

    clf.determine_sgd_step_size0(*draw_data(200))

    Tmax = 300
    eweights = np.zeros(Tmax)
    ebias = np.zeros(Tmax)

    for i in xrange(Tmax):
        X, labels = draw_data()
        # toss in some noise
        labels[0:3] = -1
        labels[3:6] = 1
        clf.partial_fit(X, labels)
        eweights[i] = 1 - np.dot(true_weights, clf.asgd_weights) / \
                (np.sqrt((true_weights ** 2).sum() * (clf.asgd_weights **
                    2).sum()))
        ebias[i] = (true_bias - clf.asgd_bias)**2

    if 0:
        import matplotlib.pyplot as plt
        plt.plot(np.arange(Tmax), eweights, label='weights cos')
        plt.plot(np.arange(Tmax), ebias, label='bias sse')
        plt.legend()
        plt.show()

    assert eweights[50:].max() < 0.010, eweights[50:].max()
    assert ebias[50:].max() < 0.010, ebias[50:].max()


@requires_theano
def run_theano_binary_asgd_speed():

    if 1:
        N_POINTS = 1000
        dtypes = ['float64']
        sizes = [1e5]
    else:
        N_POINTS = 500
        sizes=[1e2, 1e3, 1e4, 1e5]
        dtypes=['float32', 'float64']

    rstate = RandomState(42)
    XX, y = get_fake_data(N_POINTS, max(sizes), rstate)

    import theano
    dtype_orig = theano.config.floatX

    for dtype in dtypes:
        theano.config.floatX = dtype
        for N_FEATURES in sizes:
            X = XX[:,:N_FEATURES].astype(dtype)

            kwargs = dict(DEFAULT_KWARGS)
            kwargs['dtype'] = dtype

            clf0 = NaiveBinaryASGD(N_FEATURES, rstate=copy(rstate), **kwargs)
            clf1 = TheanoBinaryASGD(N_FEATURES, rstate=copy(rstate), **kwargs)
            clf1.compile_train_fn_2()

            t = time.time()
            clf0.partial_fit(X, y)
            t0 = time.time() - t

            t = time.time()
            clf1.partial_fit(X, y)
            t1 = time.time() - t
            print 'N_FEAT:%i  dtype:%s  Naive:%.3f  Theano:%.3f' % (
                    N_FEATURES, dtype, t0, t1)

    theano.config.floatX = dtype_orig
