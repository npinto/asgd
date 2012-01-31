from nose.tools import assert_equal
from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_equal

from numpy.random import RandomState
from asgd import NaiveBinaryASGD as BinaryASGD
from asgd.auto_step_size import find_sgd_step_size0
from asgd.auto_step_size import binary_fit
from asgd.auto_step_size import DEFAULT_MAX_EXAMPLES

from test_naive_asgd import get_fake_data


def get_new_model(n_features, rstate):
    return BinaryASGD(n_features, rstate=rstate,
            sgd_step_size0=1e3,
            l2_regularization=1e-3,
            n_iterations=5)


def test_binary_sgd_step_size0():
    rstate = RandomState(42)
    n_features = 20

    X, y = get_fake_data(100, n_features, rstate)

    clf = get_new_model(n_features, rstate)
    best = find_sgd_step_size0(clf, X, y, (.25, .5))
    assert_almost_equal(best, -4.9927, decimal=4)

    # start a little lower, still works
    best = find_sgd_step_size0(clf, X, y, (.125, .25))
    assert_almost_equal(best, -4.6180, decimal=4)

    # find_sgd_step_size0 does not change clf
    assert clf.sgd_step_size0 == 1000.0


def test_binary_fit():
    rstate = RandomState(42)
    n_features = 20

    clf100 = get_new_model(n_features, rstate)
    X, y = get_fake_data(100, n_features, rstate)
    _clf100 = binary_fit(clf100, X, y)
    assert _clf100 is clf100
    assert_almost_equal(clf100.sgd_step_size0, 0.04812, decimal=4)

    # smoke test
    clf1000 = get_new_model(n_features, rstate)
    X, y = get_fake_data(DEFAULT_MAX_EXAMPLES, n_features, rstate)
    _clf1000 = binary_fit(clf1000, X, y)
    assert _clf1000 is clf1000
    assert_almost_equal(clf1000.sgd_step_size0, 0.0047, decimal=4)

    # smoke test that at least it runs
    clf2000 = get_new_model(n_features, rstate)
    X, y = get_fake_data(2000, n_features, rstate)
    _clf2000 = binary_fit(clf2000, X, y)
    assert _clf2000 == clf2000
    assert_almost_equal(clf2000.sgd_step_size0, 0.0067, decimal=4)


def test_fit_replicable():

    n_features = 20

    X, y = get_fake_data(100, n_features, RandomState(4))

    m0 = get_new_model(n_features, RandomState(45))
    m0 = binary_fit(m0, X, y)

    m1 = get_new_model(n_features, RandomState(45))
    m1 = binary_fit(m1, X, y)

    assert_array_equal(m0.sgd_weights, m1.sgd_weights)
    assert_array_equal(m0.sgd_bias, m1.sgd_bias)
