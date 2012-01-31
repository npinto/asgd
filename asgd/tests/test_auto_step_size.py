import numpy as np
from numpy.random import RandomState

from asgd import NaiveBinaryASGD as BinaryASGD
from asgd.auto_step_size import binary_sgd_best_step_size
from asgd.auto_step_size import binary_fit
from asgd.auto_step_size import DEFAULT_MAX_EXAMPLES

from test_naive_asgd import get_fake_data


def test_binary_sgd_best_step_size():
    rstate = RandomState(42)
    N_FEATURES = 20

    X, y = get_fake_data(100, N_FEATURES, rstate)

    def new_model():
        return BinaryASGD(N_FEATURES, rstate=rstate,
                sgd_step_size0=1000.0,
                l2_regularization=1e-3,
                n_iterations=5)

    clf = new_model()
    x, fx, ni, nf = binary_sgd_best_step_size(clf, X, y, (.25, .5), full_output=True)

    print x, fx, ni, nf
    assert nf < 10
    assert abs(x + 5) < .5

    # start a little lower, still works
    x, fx, ni, nf = binary_sgd_best_step_size(clf, X, y, (.125, .25), full_output=True)
    assert nf < 10
    assert abs(x + 5) < .5

    # binary_sgd_best_step_size does not change clf
    assert clf.sgd_step_size0 == 1000.0


def test_binary_fit():
    rstate = RandomState(42)
    N_FEATURES = 20

    def new_model():
        return BinaryASGD(N_FEATURES, rstate=rstate,
                sgd_step_size0=1000.0,
                l2_regularization=1e-3,
                n_iterations=5)

    clf100 = new_model()
    X, y = get_fake_data(100, N_FEATURES, rstate)
    _clf100 = binary_fit(clf100, X, y)
    assert _clf100 is clf100
    assert clf100.sgd_step_size0 != 1000.0

    # smoke test
    clf1000 = new_model()
    X, y = get_fake_data(DEFAULT_MAX_EXAMPLES, N_FEATURES, rstate)
    _clf1000 = binary_fit(clf1000, X, y)
    assert _clf1000 is clf1000
    assert clf100.sgd_step_size0 != clf1000.sgd_step_size0

    # smoke test that at least it runs
    clf2000 = new_model()
    X, y = get_fake_data(2000, N_FEATURES, rstate)
    _clf2000 = binary_fit(clf2000, X, y)


def test_fit_replicable():

    N_FEATURES = 20

    def new_model():
        return BinaryASGD(N_FEATURES, rstate=RandomState(45),
                sgd_step_size0=1000.0,
                l2_regularization=1e-3,
                n_iterations=5)

    X, y = get_fake_data(100, N_FEATURES, RandomState(4))
    m0 = binary_fit(new_model(), X, y)
    m1 = binary_fit(new_model(), X, y)
    assert np.all(m0.sgd_weights == m1.sgd_weights)
    assert np.all(m0.asgd_bias == m1.asgd_bias)
