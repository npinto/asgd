"""
"""
import copy
import numpy as np
import scipy.optimize

DEFAULT_INITIAL_STEPSIZES = 0.25, 0.5
DEFAULT_MAX_EXAMPLES = 1000
DEFAULT_TOL = 0.5

def binary_sgd_best_step_size(model, X, y,
        initial_stepsizes=DEFAULT_INITIAL_STEPSIZES,
        tol=DEFAULT_TOL,
        full_output=False):
    """Use a Brent line search to find the best step size

    model - a BinaryASGD instance
    X - features
    y - labels in (-1, 1)
    initial_stepsize - (low, high) start stepsize search here
    maxiter - higher number -> more accurate minimization
    full_output - return all outputs from scipy.optimize.brent (DEBUG)

    Returns: optimal stepsize for given X, y data.
    """
    # -- stupid scipy calls some sizes twice!?
    _cache = {}
    def eval_size0(log2_size0):
        try:
            return _cache[log2_size0]
        except KeyError:
            pass
        other = copy.deepcopy(model)
        other.sgd_step_size0 = 2 ** log2_size0
        other.sgd_step_size = 2 ** log2_size0
        other.partial_fit(X, y)
        # Hack: asgd is lower variance than sgd, but it's tuned to work well
        # asymptotically, not after just a few examples
        weights = .5 * other.asgd_weights + .5 * other.sgd_weights
        bias = .5 * other.asgd_bias + .5 * other.sgd_bias

        margin = y * (np.dot(X, weights) + bias)
        l2_cost = other.l2_regularization * (weights ** 2).sum()
        rval = np.maximum(0, 1 - margin).mean() + l2_cost
        _cache[log2_size0] = rval
        return rval

    xbest = scipy.optimize.brent(
            eval_size0,
            brack=np.log2(initial_stepsizes),
            tol=tol,
            full_output=full_output)
    return xbest


def binary_fit(model, X, y,
        max_examples=DEFAULT_MAX_EXAMPLES,
        **binary_sgd_best_step_size_kwargs):
    """Returns a model with automatically-selected stepsize

    model - a BinaryASGD instance
    X - features
    y - labels in (-1, 1)
    max_examples - estimate the stepsize from at most this many examples
    initial_stepsize - (low, high) start stepsize search here

    """
    # randomly choose up to max_examples uniformly without replacement from
    # across the whole set of training data.
    idxs = model.rstate.permutation(len(X))[:max_examples]

    # Find the best learning rate for that subset
    best = binary_sgd_best_step_size(
            model,
            X[idxs],
            y[idxs],
            **binary_sgd_best_step_size_kwargs)

    # Heuristic: take the best stepsize according to the first max_examples,
    # and go half that fast for the full run.
    model.sgd_step_size0 = 2 ** (best - 1.0)
    model.sgd_step_size = 2 ** (best - 1.0)
    model.fit(X, y)
    return model

