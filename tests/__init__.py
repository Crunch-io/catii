from contextlib import contextmanager

import numpy

from catii import ccube


def arr_eq(a, b):
    """Return True if the two array-likes are close, even with NaN values."""
    return numpy.allclose(a, b, equal_nan=True)


@contextmanager
def compare_ccube_to_xcube():
    ccube._compare_output_to_xcube_for_tests = True
    try:
        yield
    finally:
        ccube._compare_output_to_xcube_for_tests = False
