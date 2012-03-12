#!/usr/bin/env python

import numpy as np
import cPickle as pkl

from asgd import NaiveBinaryASGD

print "Loading data..."
data = pkl.load(open('pf83_binary_0.pkl'))

Xtrn = data['Xtrn']
Xtst = data['Xtst']
Ytrn = data['Ytrn']
Ytst = data['Ytst']

n_samples, n_features = Xtrn.shape

svm = NaiveBinaryASGD(n_features, l2_regularization=1e-2)

np.random.seed(42)

for i in xrange(100):

    print "fit..."
    print "iteration", (i + 1)
    ridx = np.random.permutation(n_samples)
    Xtrn = Xtrn[ridx]
    Ytrn = Ytrn[ridx]
    svm.partial_fit(Xtrn, Ytrn)

    print "predict..."
    svm.predict(Xtst)
    gv = np.array(svm.predict(Xtst))

    pos = (Ytst > 0)
    neg = - pos
    bacc = .5 * ((Ytst[pos] == gv[pos]).mean() + (Ytst[neg] == gv[neg]).mean())

    print 'balanced accuracy', bacc
