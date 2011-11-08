from nose.tools import assert_equal
import numpy as np

from asgd import NaiveMulticlassASGD as ASGD


def get_fake_data(n_points, n_features, rseed):
    np.random.seed(rseed)
    X = np.random.randn(n_points, n_features).astype('f')
    y = (np.random.randn(n_points)>0)
    X[y == 1] += 1e-1
    return X, y

def get_fake_multiclass_data(n_points, n_features, n_classes, rseed):
    np.random.seed(rseed)
    X = np.random.randn(n_points, n_features).astype('f')
    z = np.random.random((n_points,))
    y = np.zeros((n_points,))
    for ind in range(n_classes):
        y[(z < float(ind+1)/n_classes) & (z >= float(ind)/n_classes)] = ind
    return X, y

def test_naive_asgd():
    n_points = 1e3
    n_features = 1e2
    X, y = get_fake_data(n_points, n_features, 42)
    Xtst, ytst = get_fake_data(n_points, n_features, 43)

    clf = ASGD(n_features, sgd_step_size0=1e-3, l2_regularization=1e-6,
               n_iterations=4, dtype=np.float32)
    clf.fit(X, y)
    ytrn_preds = clf.predict(X)
    ytst_preds = clf.predict(Xtst)
    ytrn_acc = (ytrn_preds==y).mean()
    ytst_acc = (ytst_preds==y).mean()
    assert_equal(ytrn_acc, 0.723)
    assert_equal(ytst_acc, 0.513)


def test_naive_asgd_with_feedback():
    n_points = 1e3
    n_features = 1e2
    X, y = get_fake_data(n_points, n_features, 42)
    Xtst, ytst = get_fake_data(n_points, n_features, 43)

    clf = ASGD(n_features, sgd_step_size0=1e-3, l2_regularization=1e-6,
               n_iterations=4, feedback=True, dtype=np.float32)
    clf.fit(X, y)
    ytrn_preds = clf.predict(X)
    ytst_preds = clf.predict(Xtst)
    ytrn_acc = (ytrn_preds==y).mean()
    ytst_acc = (ytst_preds==y).mean()
    assert_equal(ytrn_acc, 0.707)
    assert_equal(ytst_acc, 0.505)


def test_naive_multiclass_asgd():
    n_points = 1e3
    n_features = 1e2
    X, y = get_fake_multiclass_data(n_points, n_features, 3, 42)
    Xtst, ytst = get_fake_multiclass_data(n_points, n_features, 3, 43)

    clf = ASGD(n_features, sgd_step_size0=1e-3, l2_regularization=1e-6,
               n_iterations=4, dtype=np.float32, n_classes=3)
    clf.fit(X, y)
    ytrn_preds = clf.predict(X)
    ytst_preds = clf.predict(Xtst)
    ytrn_acc = (ytrn_preds==y).mean()
    ytst_acc = (ytst_preds==y).mean()
    assert_equal(ytrn_acc, 0.511)
    assert_equal(ytst_acc, 0.324)
