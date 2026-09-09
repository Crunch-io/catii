"""Tests for xfunc_stddev with extreme values that can cause overflow."""

import math

import numpy
import pytest

from catii import xcube, xfuncs


class TestXfuncStddevOverflow:
    """Test that stddev handles extreme values without producing inf."""

    def test_stddev_with_extreme_value_no_dimensions(self):
        """
        When a numeric array contains an extremely large value (1.6018e+188),
        the stddev calculation should not produce inf.

        The squared_variances = (summables - wmeans) ** 2 line can overflow
        to inf when the value is extremely large. This should be handled
        gracefully by returning nan instead of inf.
        """
        # Simulate a distribution similar to the real-world case:
        # - Most values are 0.0
        # - One extreme value that causes overflow when squared
        values = (
            [0.0] * 100
            + [60.0] * 5
            + [50.0] * 5
            + [100.0] * 3
            + [1.6018e188]  # Extreme value that causes overflow
        )
        factvar = numpy.array(values)

        result = xcube([]).stddev(factvar)

        # Result should be nan (due to the extreme value), not inf
        # inf would cause JSON serialization errors downstream
        assert not numpy.isinf(result), f"stddev returned inf: {result}"

    def test_stddev_with_extreme_value_one_dimension(self):
        """
        When computing stddev across a categorical dimension, cells containing
        extreme values should return nan, not inf.
        """
        # Create a dimension with two categories
        arr1 = [0] * 57 + [1] * 57  # 57 items in each category

        # Values where category 1 contains the extreme value
        values = (
            # Category 0: normal values
            [0.0] * 50 + [60.0] * 5 + [100.0] * 2
            # Category 1: includes extreme value
            + [0.0] * 50 + [60.0] * 5 + [100.0] * 1 + [1.6018e188]
        )
        factvar = numpy.array(values)

        result = xcube([arr1]).stddev(factvar)

        # Category 0 should have a valid stddev
        assert not numpy.isnan(result[0]), f"Category 0 should have valid stddev: {result}"
        assert not numpy.isinf(result[0]), f"Category 0 should not be inf: {result}"

        # Category 1 should be nan (not inf) due to the extreme value
        assert not numpy.isinf(result[1]), f"Category 1 returned inf instead of nan: {result}"

    def test_stddev_with_extreme_value_weighted(self):
        """
        Weighted stddev should also handle extreme values gracefully.
        """
        values = [0.0] * 10 + [1.6018e188]
        weights = [1.0] * 11
        factvar = numpy.array(values)

        result = xcube([]).stddev(factvar, weights=numpy.array(weights))

        assert not numpy.isinf(result), f"Weighted stddev returned inf: {result}"

    def test_stddev_overflow_in_squared_variance(self):
        """
        Direct test: when (value - mean) ** 2 overflows, the result should
        be nan, not inf.

        For a value like 1e200, subtracting a small mean and squaring will
        overflow float64, producing inf. This inf should be converted to nan.
        """
        # Use a value that will definitely overflow when squared
        extreme_value = 1e200
        values = numpy.array([0.0, 1.0, 2.0, extreme_value])

        result = xcube([]).stddev(values)

        assert not numpy.isinf(result), (
            f"stddev produced inf from overflow. "
            f"Value {extreme_value} caused (x - mean)^2 to overflow. "
            f"Result: {result}"
        )
