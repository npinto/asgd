"""
BinaryASGD - this symbol refers to the fastest available implementation of the
BinaryASGD algorithm, as defined in naive_asgd.

OVAASGD - this symbol refers to the fastest available implementation of the
OVAASGD algorithm, as defined in naive_asgd.
"""

# naive_asgd defines reference implementations
from naive_asgd import NaiveBinaryASGD, NaiveOVAASGD
BinaryASGD = NaiveBinaryASGD
NaiveOVAASGD = NaiveOVAASGD

# theano_asgd requires theano, provides faster implementations than naive_asgd.
try:
    from theano_asgd import TheanoBinaryASGD
    BinaryASGD = TheanoBinaryASGD
except ImportError:
    pass

from experimental_asgd import ExperimentalBinaryASGD
