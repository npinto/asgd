import copy
import numpy as np
from scipy import optimize

DEFAULT_INITIAL_RANGE = 0.25, 0.5
DEFAULT_MAX_EXAMPLES = 1000
DEFAULT_TOLERANCE = 0.5
DEFAULT_BRENT_OUTPUT = False


def find_sgd_step_size0(
    model, X, y,
    initial_range=DEFAULT_INITIAL_RANGE,
    tolerance=DEFAULT_TOLERANCE, brent_output=DEFAULT_BRENT_OUTPUT):
    """Use a Brent line search to find the best step size

    Parameters
    ----------
    model: BinaryASGD
        Instance of a BinaryASGD

    X: array, shape = [n_samples, n_features]
        Array of features

    y: array, shape = [n_samples]
        Array of labels in (-1, 1)

    initial_range: tuple of float
        Initial range for the sgd_step_size0 search (low, high)

    max_iterations:
        Maximum number of interations

    Returns
    -------
    best_sgd_step_size0: float
        Optimal sgd_step_size0 given `X` and `y`.
    """
    # -- stupid scipy calls some sizes twice!?
    _cache = {}

    def eval_size0(log2_size0):
        try:
            return _cache[log2_size0]
        except KeyError:
            pass
        other = copy.deepcopy(model)
        current_step_size = 2 ** log2_size0
        other.sgd_step_size0 = current_step_size
        other.sgd_step_size = current_step_size
        other.partial_fit(X, y)
        # Hack: asgd is lower variance than sgd, but it's tuned to work
        # well asymptotically, not after just a few examples
        weights = .5 * (other.asgd_weights + other.sgd_weights)
        bias = .5 * (other.asgd_bias + other.sgd_bias)

        margin = y * (np.dot(X, weights) + bias)
        l2_cost = other.l2_regularization * (weights ** 2).sum()
        rval = np.maximum(0, 1 - margin).mean() + l2_cost
        _cache[log2_size0] = rval
        return rval

    best_sgd_step_size0 = optimize.brent(
        eval_size0, brack=np.log2(initial_range), tol=tolerance)

    return best_sgd_step_size0


def binary_fit(
    model, X, y,
    max_examples=DEFAULT_MAX_EXAMPLES,
    **find_sgd_step_size0_kwargs):
    """Returns a model with automatically-selected sgd_step_size0

    Parameters
    ----------
    model: BinaryASGD
        Instance of the model to be fitted.

    X: array, shape = [n_samples, n_features]
        Array of features

    y: array, shape = [n_samples]
        Array of labels in (-1, 1)

    max_examples: int
        Maximum number of examples to use from `X` and `y` to find an
        estimate of the best sgd_step_size0

    Returns
    -------
    model: BinaryASGD
        Instances of the model, fitted with an estimate of the best
        sgd_step_size0
    """

    assert X.ndim == 2
    assert len(X) == len(y)
    assert max_examples > 0

    # randomly choose up to max_examples uniformly without replacement from
    # across the whole set of training data.
    idxs = model.rstate.permutation(len(X))[:max_examples]

    # Find the best learning rate for that subset
    best = find_sgd_step_size0(
        model, X[idxs], y[idxs], **find_sgd_step_size0_kwargs)

    # Heuristic: take the best stepsize according to the first max_examples,
    # and go half that fast for the full run.
    best_estimate = 2. ** (best - 1.0)
    model.sgd_step_size0 = best_estimate
    model.sgd_step_size = best_estimate
    model.fit(X, y)

    return model
