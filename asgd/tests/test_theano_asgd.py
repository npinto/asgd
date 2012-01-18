import time
from copy import copy

from nose.tools import assert_equal, raises
from nose.plugins.skip import SkipTest
from numpy.testing import assert_allclose
import numpy as np
from numpy.random import RandomState

from asgd.theano_asgd import TheanoBinaryASGD
from asgd.naive_asgd import NaiveBinaryASGD
from test_naive_asgd import get_fake_data

RTOL = 1e-6
ATOL = 1e-6

N_POINTS = 1e3
N_FEATURES = 1e2

DEFAULT_ARGS = (N_FEATURES,)
DEFAULT_KWARGS = dict(sgd_step_size0=1e-3,
                      l2_regularization=1e-6,
                      n_iterations=4,
                      dtype=np.float32)



def test_theano_binary_asgd_like_naive_asgd():

    rstate = RandomState(42)

    X, y = get_fake_data(N_POINTS, N_FEATURES, rstate)
    Xtst, ytst = get_fake_data(N_POINTS, N_FEATURES, rstate)

    clf0 = NaiveBinaryASGD(*DEFAULT_ARGS, rstate=copy(rstate), **DEFAULT_KWARGS)
    clf1 = TheanoBinaryASGD(*DEFAULT_ARGS, rstate=copy(rstate), **DEFAULT_KWARGS)

    for clf in [clf0, clf1]:
        clf.fit(X, y)
        ytrn_preds = clf.predict(X)
        ytst_preds = clf.predict(Xtst)
        ytrn_acc = (ytrn_preds == y).mean()
        ytst_acc = (ytst_preds == y).mean()
        assert_equal(ytrn_acc, 0.72)
        assert_equal(ytst_acc, 0.522)


def test_theano_binary_asgd_converges_to_truth():
    n_features = 5

    rstate = RandomState(42)

    true_weights = rstate.randn(n_features)
    true_bias = rstate.randn() * 0
    clf = TheanoBinaryASGD(n_features,
            rstate=rstate,
            sgd_step_size0=0.1,
            dtype='float64',
            l2_regularization = 1e-4,
            sgd_step_size_scheduling_exponent=0.5,
            sgd_step_size_scheduling_multiplier=1.0)

    Tmax = 300

    eweights = np.zeros(Tmax)
    ebias = np.zeros(Tmax)

    for i in xrange(Tmax):
        X = rstate.randn(20, 5)
        labels = np.sign(np.dot(X, true_weights) + true_bias).astype('int')
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
    else:
        assert eweights[50:].max() < 0.002
        assert ebias[50:].max() < 0.005


def run_theano_binary_asgd_speed():

    rstate = RandomState(42)

    N_FEATURES = 10000

    X, y = get_fake_data(N_POINTS, N_FEATURES, rstate)
    Xtst, ytst = get_fake_data(N_POINTS, N_FEATURES, rstate)

    kwargs = dict(DEFAULT_KWARGS)
    kwargs['n_iterations'] = 10
    kwargs['dtype'] = 'float32'

    clf0 = NaiveBinaryASGD(N_FEATURES, rstate=copy(rstate), **DEFAULT_KWARGS)
    clf1 = TheanoBinaryASGD(N_FEATURES, rstate=copy(rstate), **DEFAULT_KWARGS)
    clf1.compile_train_fn_2()

    t = time.time()
    clf0.fit(X, y)
    t0 = time.time() - t

    t = time.time()
    clf1.fit(X, y)
    t1 = time.time() - t
    print 'TIMES', 'Naive', t0, 'Theano', t1

