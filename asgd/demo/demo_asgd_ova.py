#!/usr/bin/env python

STRIDE = 100
N_ITERATIONS = 10

import numpy as np
import time

from asgd import NaiveOVAASGD

print "Loading data..."
import load_data
X_trn = load_data.X_trn()[::STRIDE]
y_trn = load_data.y_trn()[::STRIDE]
X_tst = load_data.X_tst()[::STRIDE]
y_tst = load_data.y_tst()[::STRIDE]

n_samples, n_features = X_trn.shape
classes = np.unique(y_trn)
n_classes = len(classes)

clf = NaiveOVAASGD(n_classes, n_features, l2_regularization=1e-2)

np.random.seed(42)
tot_time = 0
for i in xrange(N_ITERATIONS):
    print "*" * 80
    print ">>> Iteration %d" % (i + 1)
    print "> randomize..."
    ridx = np.random.permutation(n_samples)
    X_trn = X_trn[ridx].copy()
    y_trn = y_trn[ridx].copy()
    print "> partial_fit..."
    start = time.time()
    clf.partial_fit(X_trn, y_trn)
    end = time.time()
    i_time = (end - start)
    print "> partial_fit time: %f" % i_time
    tot_time += i_time

    print "> predict..."
    clf.predict(X_tst)
    y_pred = np.array(clf.predict(X_tst))

    acc = (y_tst == y_pred).mean()
    print "> accuracy=%f, tot_time=%f" % (acc, tot_time)
