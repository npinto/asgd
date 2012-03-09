import numpy as np
import cPickle as pkl

#from scikits_libsvm_one_vs_all import ScikitsMultiClassSVM 
from asgd import NaiveBinaryASGD

data = pkl.load(open('pf83_binary_0.pkl'))

Xtrn = data['Xtrn']
Xtst = data['Xtst']
Ytrn = data['Ytrn']
Ytst = data['Ytst']

#svm = ScikitsMultiClassSVM(C=1e5)
svm = NaiveBinaryASGD(Xtrn.shape[1], l2_regularization=1e-6)
svm.fit(Xtrn, Ytrn)
svm.predict(Xtst)
gv = np.array(svm.predict(Xtst))

pos = (Ytst > 0)
neg = - pos
bacc = .5 * ((Ytst[pos] == gv[pos]).mean() + (Ytst[neg] == gv[neg]).mean())

print bacc
