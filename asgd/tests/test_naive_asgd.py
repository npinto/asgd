from nose.tools import assert_equal
from numpy.testing import assert_allclose
import numpy as np

from asgd import NaiveBinaryASGD as ASGD
from asgd import NaiveOVAASGD as OVAASGD

RTOL = 1e-6
ATOL = 1e-6

N_POINTS = 1e3
N_FEATURES = 1e2

def get_fake_data(n_points, n_features, rseed):
    np.random.seed(rseed)
    X = np.random.randn(n_points, n_features).astype(np.float32)
    y = 2 * (np.random.randn(n_points) > 0) - 1
    X[y == 1] += 1e-1
    return X, y


def get_fake_binary_data_multi_labels(n_points, n_features, rseed):
    np.random.seed(rseed)
    X = np.random.randn(n_points, n_features).astype(np.float32)
    y = np.random.randn(n_points) > 0
    X[y] += 1e-1
    return X, y


def get_fake_multiclass_data(n_points, n_features, n_classes, rseed):
    np.random.seed(rseed)
    X = np.random.randn(n_points, n_features).astype(np.float32)
    z = np.random.random((n_points,))
    y = np.zeros((n_points,))
    for ind in range(n_classes):
        I = (z < float(ind + 1) / n_classes) & (z >= float(ind) / n_classes)
        y[I] = ind
    return X, y


def test_naive_asgd():
    X, y = get_fake_data(N_POINTS, N_FEATURES, 42)
    Xtst, ytst = get_fake_data(N_POINTS, N_FEATURES, 43)

    clf = ASGD(N_FEATURES, sgd_step_size0=1e-3, l2_regularization=1e-6,
               n_iterations=4, dtype=np.float32)
    clf.fit(X, y)
    ytrn_preds = clf.predict(X)
    ytst_preds = clf.predict(Xtst)
    ytrn_acc = (ytrn_preds == y).mean()
    ytst_acc = (ytst_preds == y).mean()
    assert_equal(ytrn_acc, 0.723)
    assert_equal(ytst_acc, 0.513)


def test_naive_asgd_with_feedback():
    X, y = get_fake_data(N_POINTS, N_FEATURES, 42)
    Xtst, ytst = get_fake_data(N_POINTS, N_FEATURES, 43)

    clf = ASGD(N_FEATURES, sgd_step_size0=1e-3, l2_regularization=1e-6,
               n_iterations=4, feedback=True, dtype=np.float32)
    clf.fit(X, y)
    ytrn_preds = clf.predict(X)
    ytst_preds = clf.predict(Xtst)
    ytrn_acc = (ytrn_preds == y).mean()
    ytst_acc = (ytst_preds == y).mean()
    assert_equal(ytrn_acc, 0.707)
    assert_equal(ytst_acc, 0.505)


def test_naive_asgd_multi_labels():
    X, y = get_fake_binary_data_multi_labels(N_POINTS, N_FEATURES, 42)
    Xtst, ytst = get_fake_binary_data_multi_labels(N_POINTS, N_FEATURES, 43)

    # n_classes is 2 since it is actually a binary case
    clf = OVAASGD(2, N_FEATURES, sgd_step_size0=1e-3,
                  l2_regularization=1e-6, n_iterations=4, dtype=np.float32)
    clf.fit(X, y)
    ytrn_preds = clf.predict(X)
    ytst_preds = clf.predict(Xtst)
    ytrn_acc = (ytrn_preds == y).mean()
    ytst_acc = (ytst_preds == y).mean()
    assert_equal(ytrn_acc, 0.723)
    assert_equal(ytst_acc, 0.513)


def test_naive_multiclass_ova_asgd():

    n_classes = 10

    X, y = get_fake_multiclass_data(N_POINTS, N_FEATURES, n_classes, 42)
    Xtst, ytst = get_fake_multiclass_data(N_POINTS, N_FEATURES, n_classes, 43)

    clf = OVAASGD(n_classes, N_FEATURES, sgd_step_size0=1e-3,
                  l2_regularization=1e-6, n_iterations=4, dtype=np.float32)
    clf.fit(X, y)
    ytrn_preds = clf.predict(X)
    ytst_preds = clf.predict(Xtst)
    ytrn_acc = (ytrn_preds == y).mean()
    ytst_acc = (ytst_preds == y).mean()
    assert_equal(ytrn_acc, 0.364)
    assert_equal(ytst_acc, 0.116)


def test_naive_multiclass_ova_vs_binary_asgd():

    n_classes = 3

    Xtrn, ytrn = get_fake_multiclass_data(N_POINTS, N_FEATURES, n_classes, 42)
    Xtst, ytst = get_fake_multiclass_data(N_POINTS, N_FEATURES, n_classes, 43)

    args = (N_FEATURES,)
    kwargs = dict(sgd_step_size0=1e-3,
                  l2_regularization=1e-6,
                  n_iterations=4,
                  dtype=np.float32)

    # -- ground truth 'gt'
    # emulate OVA with binary asgd classifiers
    ytrn0 = 2 * (ytrn == 0).astype(np.int) - 1
    ytrn1 = 2 * (ytrn == 1).astype(np.int) - 1
    ytrn2 = 2 * (ytrn == 2).astype(np.int) - 1

    clf0 = ASGD(*args, **kwargs)
    clf0.partial_fit(Xtrn, ytrn0)
    clf1 = ASGD(*args, **kwargs)
    clf1.partial_fit(Xtrn, ytrn1)
    clf2 = ASGD(*args, **kwargs)
    clf2.partial_fit(Xtrn, ytrn2)

    m0 = clf0.decision_function(Xtst)
    m1 = clf1.decision_function(Xtst)
    m2 = clf2.decision_function(Xtst)

    gt = np.column_stack([m0, m1, m2])

    # -- given 'gv'
    clf = OVAASGD(*((n_classes,) + args), **kwargs)
    clf.partial_fit(Xtrn, ytrn)
    gv = clf.decision_function(Xtst)

    assert_allclose(gv, gt, rtol=RTOL, atol=ATOL)
