"""Averaging Stochastic Gradient Descent Classifier
"""

import numpy as np
from itertools import izip
from numpy import dot, add, multiply, subtract


class ExperimentalBinaryASGD(object):

    def __init__(self, n_features, sgd_step_size0=1, l2_regularization=0,
                 max_iter=np.inf,
                 dtype=np.float32):

        self.n_features = n_features
        self.max_iter = max_iter

        assert l2_regularization > 0
        self.l2_regularization = l2_regularization
        self.dtype = dtype

        self.sgd_weights = np.zeros((n_features), dtype=dtype)
        self.sgd_bias = np.zeros((1), dtype=dtype)
        self.sgd_step_size0 = sgd_step_size0
        self.sgd_step_size_scheduling_exponent = 2. / 3
        self.sgd_step_size_pos = sgd_step_size0
        self.sgd_step_size_neg = sgd_step_size0

        self.asgd_weights = np.zeros((n_features), dtype=dtype)
        self.asgd_bias = np.zeros((1), dtype=dtype)
        self.asgd_step_size0 = 1
        self.asgd_step_size = self.asgd_step_size0

        self.n_iterations = 0
        self.n_iterations_pos = 0
        self.n_iterations_neg = 0
        self.margin = np.empty((1), dtype=dtype)

        try:
            self.learn_once = profile(self.learn_once)
        except NameError:
            pass

    def learn_once(self, data_iterator, label_iterator):

        sgd_step_size0 = self.sgd_step_size0
        sgd_step_size_scheduling_exponent = self.sgd_step_size_scheduling_exponent
        sgd_step_size_pos = self.sgd_step_size_pos
        sgd_step_size_neg = self.sgd_step_size_neg
        sgd_weights = self.sgd_weights
        sgd_bias = self.sgd_bias

        asgd_weights = self.asgd_weights
        asgd_bias = self.asgd_bias
        asgd_step_size = self.asgd_step_size

        l2_regularization = self.l2_regularization

        n_iterations = self.n_iterations
        n_iterations_pos = self.n_iterations_pos
        n_iterations_neg = self.n_iterations_neg
        margin = self.margin

        sgd_step_size_by_label = np.empty((1), dtype=self.dtype)
        sgd_step_size_by_label_by_data = np.empty((self.n_features), dtype=self.dtype)
        one_minus_asgd_step_size = np.empty((1), dtype=self.dtype)
        #one = np.ones((1), dtype=self.dtype)
        asgd_step_size_by_sgd_weights = np.empty_like(sgd_weights)
        from numpy import linalg

        #good = 0
        n_l = 1
        avg_loss = 0
        loss_step_size = 1.
        active_set = []
        for di, (data, label) in enumerate(izip(data_iterator, label_iterator)):

            # 1. compute margin
            # naive code: margin = label * (dot(data, sgd_weights) + sgd_bias)
            dec = dot(data, sgd_weights)
            add(dec, sgd_bias, margin)
            multiply(label, margin, margin)

            # 2.2 update sgd
            current_loss = 0

            if label > 0:
                sgd_step_size = sgd_step_size_pos
                n_iterations = n_iterations_pos
            else:
                sgd_step_size = sgd_step_size_neg
                n_iterations = n_iterations_neg
            #print sgd_step_size, label, n_iterations

            if l2_regularization:
                sgd_weights *= (1 - l2_regularization * sgd_step_size)
                current_loss += (l2_regularization / 2.) * linalg.norm(sgd_weights)

            if margin < 1:

                # naive code: sgd_weights += sgd_step_size * label * data
                multiply(sgd_step_size, label, sgd_step_size_by_label)
                multiply(sgd_step_size_by_label, data, sgd_step_size_by_label_by_data)
                add(sgd_weights, sgd_step_size_by_label_by_data, sgd_weights)

                # naive_code: sgd_bias += sgd_step_size * label
                add(sgd_bias, sgd_step_size_by_label, sgd_bias)

                current_loss += 1 - margin
                active_set += [di]

            avg_loss = (1. - loss_step_size) * avg_loss + loss_step_size * current_loss
            n_l += 1
            loss_step_size = 1. / n_l

            # 2.2 update asgd

            # naive code: asgd_weights = (1 - asgd_step_size) * asgd_weights + asgd_step_size * sgd_weights
            one_minus_asgd_step_size = 1 - asgd_step_size
            multiply(asgd_weights, one_minus_asgd_step_size, asgd_weights)
            multiply(asgd_step_size, sgd_weights, asgd_step_size_by_sgd_weights)
            add(asgd_weights, asgd_step_size_by_sgd_weights, asgd_weights)

            # naive code: asgd_bias = (1 - asgd_step_size) * asgd_bias + asgd_step_size * sgd_bias
            multiply(asgd_bias, one_minus_asgd_step_size, asgd_bias)
            #asgd_bias += asgd_step_size * sgd_bias
            asgd_step_size_by_sgd_bias = asgd_step_size * sgd_bias
            #add(asgd_bias, asgd_step_size * sgd_bias, asgd_bias)
            add(asgd_bias, asgd_step_size_by_sgd_bias, asgd_bias)

            # 4.1 update step_sizes
            n_iterations += 1

            sgd_step_size = sgd_step_size0 / ((1 + l2_regularization * sgd_step_size0 * n_iterations) ** sgd_step_size_scheduling_exponent)
            asgd_step_size = 1. / n_iterations

            #print 'sgd_step_size:', self.sgd_step_size_pos, self.sgd_step_size_neg

            #asgd_step_size = 1. / min(n_iterations, 1e3)
            #asgd_step_size = 1e-3#1. / min(n_iterations, 1e3)
            #print sgd_step_size, asgd_step_size
            #print n_iterations
            #print margin
            if label > 0:
                sgd_step_size_pos = sgd_step_size
                n_iterations_pos = n_iterations
            else:
                sgd_step_size_neg = sgd_step_size
                n_iterations_neg = n_iterations


        # --
        print 'n_iterations', n_iterations_pos, n_iterations_neg, n_iterations_pos + n_iterations_neg
        #print good
        print 'running average loss:', avg_loss

        self.sgd_weights = sgd_weights
        self.sgd_bias = sgd_bias
        self.sgd_step_size_pos = sgd_step_size_pos
        self.sgd_step_size_neg = sgd_step_size_neg

        self.asgd_weights = asgd_weights
        self.asgd_bias = asgd_bias
        self.asgd_step_size = asgd_step_size

        self.n_iterations_pos = n_iterations_pos
        self.n_iterations_neg = n_iterations_neg
        self.n_iterations = n_iterations_pos + n_iterations_neg

        return avg_loss, np.array(active_set)

    def fit(self, X, y):
        last_accuracy = -1
        patience = 1
        n_iter = 0
        max_iter = self.max_iter
        n_points = X.shape[0]

        print 'train asgd'

        if 0:
            validation_fraction = 2e-1
            val_idx = np.random.permutation(n_points)[:int(validation_fraction*n_points)]
            print 'val_idx', val_idx.shape
            Xval = X[val_idx]
            yval = y[val_idx]
            Xtrn = X[-val_idx]
            ytrn = y[-val_idx]
        else:
            Xval = X
            yval = y
            Xtrn = X
            ytrn = y

        done = False
        min_loss = np.inf
        best_asgd_weights = self.asgd_weights.copy()
        best_asgd_bias = self.asgd_bias.copy()
        best_active_set = []
        while True:

            for _ in xrange(patience):

                idx = np.random.permutation(Xtrn.shape[0])
                Xb = Xtrn[idx]
                yb = ytrn[idx]
                current_loss, current_active_set = self.learn_once(Xb, yb)
                n_iter += 1

                print "current_loss:", current_loss
                print "min_loss:", min_loss
                print "len(active_set):", len(current_active_set)

                same_set = sorted(best_active_set) == sorted(current_active_set)
                if len(current_active_set) == 0 or (same_set and current_loss > min_loss):
                    done = True
                    break
                else:
                    min_loss = current_loss
                    best_asgd_weights = self.asgd_weights
                    best_asgd_bias = self.asgd_bias
                    best_active_set = current_active_set

                # replace Xtrn by the active set 
                Xtrn = Xtrn[idx[current_active_set]]

                #self.learn_once(X, y)
                #sgd_preds = np.sign(dot(asgd.sgd_weights, Xb.T) + asgd.sgd_bias)
                #print 'sgd', (sgd_preds == yb).mean()
                if n_iter >= max_iter:
                    break

            if done:
                break

            asgd_preds = np.sign(dot(self.asgd_weights, Xval.T) + self.asgd_bias)
            current_accuracy = (asgd_preds == yval).mean()
            print 'asgd', (asgd_preds == yval).mean()
            sgd_preds = np.sign(dot(self.sgd_weights, Xval.T) + self.sgd_bias)
            current_accuracy = (sgd_preds == yval).mean()
            print 'sgd', (sgd_preds == yval).mean()

            print patience, current_accuracy, last_accuracy

            if current_accuracy > last_accuracy:
                last_weights = self.asgd_weights
                last_bias = self.asgd_bias
                last_accuracy = current_accuracy
                patience = int(patience * 2)
            else:
                best_asgd_weights = last_weights
                best_asgd_bias = last_bias
                break

            if n_iter > max_iter:
                break

        self.asgd_weights = best_asgd_weights
        self.asgd_bias = best_asgd_bias
        print 'n_iter batch:', n_iter

    def predict(self, X):
        return np.sign(dot(self.asgd_weights, X.T) + self.asgd_bias)
