# Simple Demo of LIBSVM Classification with L3 Features
# Zak Stone
# zstone@gmail.com
# 2011

# NOTE: This demo will only run quickly if NumPy is connected
# to a decent BLAS implementation! The all-in-one Enthought
# Python Distribution (free for academics) is probably the
# easiest starting point:
# http://enthought.com/

import numpy as np
from scipy import io
from collections import defaultdict
from time import time

def run_libsvm(mat_path):
    print "Loading %s..." % mat_path
    mat_dict = io.loadmat(mat_path)
    test_svm(mat_dict['X'], mat_dict['Y'], 90, 10)

def test_svm(descs, labels, num_train, num_test):
    X = []
    X_test = []
    Y = []
    Y_test = []
    counts = defaultdict(int)

    # split the data deterministically for
    # demonstration purposes
    for i, label in enumerate(labels.ravel()):
        row = descs[i,:]
        if counts[label] < num_train:
            X.append(row)
            Y.append(label)
        elif counts[label] < num_train + num_test:
            X_test.append(row)
            Y_test.append(label)
        else:
            continue

        counts[label] += 1

    # not a reasonable approach for large X
    X = np.vstack(X)
    X_test = np.vstack(X_test)

    print '-'*20
    print 'Training one-versus-all SVMs...'
    t0 = time()
    import IPython; ipshell = IPython.embed; ipshell(banner1='ipshell')
    #svm = ScikitsMultiClassSVM(C=1e5)

    ## NOTE: This fit routine will destructively modify its arguments;
    ## hence the copies
    #svm.fit(X.copy(), Y)
    #Y_pred = svm.predict(X_test.copy())

    #dt1 = time() - t0
    #correct1 = sum([1 if y_pred == y_test else 0 for y_pred, y_test in zip(Y_pred, Y_test)])

    #print "LIBSVM: %s correct out of %s;\x1b[32m %s \x1b[mpercent in\x1b[34m %f \x1b[ms" % (correct1, len(Y_test), float(correct1)/len(Y_test)*100, dt1)

if __name__ == '__main__':
    run_libsvm('pubfig83_descs.mat')
