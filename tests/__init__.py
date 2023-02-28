import numpy


def arr_eq(a, b):
    """Return True if the two array-likes are close, even with NaN values."""
    return numpy.allclose(a, b, equal_nan=True)
