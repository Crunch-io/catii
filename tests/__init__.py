import numpy

from catii import ccube


def arr_eq(a, b):
    """Return True if the two array-likes are close, even with NaN values."""
    return numpy.allclose(a, b, equal_nan=True)


def compare_ccube_to_xcube(f):
    def _with_cube_comparison(*args, **kwargs):
        ccube._compare_output_to_xcube_for_tests = True
        try:
            return f(*args, **kwargs)
        finally:
            ccube._compare_output_to_xcube_for_tests = False

    return _with_cube_comparison
