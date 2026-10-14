"""Tests for xfunc_numeric_summary.

This xfunc computes multiple numeric statistics in a single pass:
- mean
- stddev
- valid_count (unweighted count of non-missing values)
- sum
- quantiles (configurable list of probabilities)

The goal is to reduce both:
1. The number of cube calls from cr.lib
2. The number of calculations within catii by sharing intermediates
"""

import operator
from functools import reduce

import numpy
import pytest

from catii import xcube, xfuncs

from .. import arr_eq


class TestNumericSummaryBasicInterface:
    """Test the basic interface and configuration of numeric_summary."""

    def test_default_includes_all_statistics(self):
        """By default, all statistics should be computed."""
        arr = [1.0, 2.0, 3.0, 4.0, 5.0]
        f = xfuncs.xfunc_numeric_summary(arr)

        cube = xcube([])
        result = f.calculate(cube)

        # Result should be a dict-like object with all statistics
        assert "mean" in result
        assert "stddev" in result
        assert "valid_count" in result
        assert "sum" in result
        assert "quantiles" in result

        # Check values
        assert result["mean"] == pytest.approx(3.0)
        assert result["sum"] == pytest.approx(15.0)
        assert result["valid_count"] == 5
        # stddev of [1,2,3,4,5] with ddof=1 is sqrt(2.5) ≈ 1.5811
        assert result["stddev"] == pytest.approx(numpy.std(arr, ddof=1))

        # Default quantiles should be: 0, 0.25, 0.5, 0.75, 1.0
        assert len(result["quantiles"]) == 5
        assert result["quantiles"][0.0] == pytest.approx(1.0)  # min
        assert result["quantiles"][0.25] == pytest.approx(2.0)  # Q1
        assert result["quantiles"][0.5] == pytest.approx(3.0)  # median
        assert result["quantiles"][0.75] == pytest.approx(4.0)  # Q3
        assert result["quantiles"][1.0] == pytest.approx(5.0)  # max

    def test_configure_subset_of_statistics(self):
        """Should be able to request only specific statistics."""
        arr = [1.0, 2.0, 3.0, 4.0, 5.0]
        f = xfuncs.xfunc_numeric_summary(
            arr,
            include={"mean": True, "stddev": True}
        )

        cube = xcube([])
        result = f.calculate(cube)

        assert "mean" in result
        assert "stddev" in result
        assert "valid_count" not in result
        assert "sum" not in result
        assert "quantiles" not in result

    def test_configure_specific_quantiles(self):
        """Should be able to specify which quantiles to compute."""
        arr = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        f = xfuncs.xfunc_numeric_summary(
            arr,
            include={"quantiles": [0.1, 0.5, 0.9]}
        )

        cube = xcube([])
        result = f.calculate(cube)

        assert "quantiles" in result
        assert len(result["quantiles"]) == 3
        assert 0.1 in result["quantiles"]
        assert 0.5 in result["quantiles"]
        assert 0.9 in result["quantiles"]
        assert result["quantiles"][0.5] == pytest.approx(5.5)  # median

    def test_empty_include_raises_error(self):
        """Requesting no statistics should raise an error."""
        arr = [1.0, 2.0, 3.0]
        with pytest.raises(ValueError, match="at least one statistic"):
            xfuncs.xfunc_numeric_summary(arr, include={})


class TestNumericSummaryWithMissing:
    """Test handling of missing values."""

    def test_missing_values_with_nan(self):
        """NaN values should be treated as missing."""
        arr = [1.0, 2.0, float("nan"), 4.0, 5.0]
        f = xfuncs.xfunc_numeric_summary(arr)

        cube = xcube([])
        result = f.calculate(cube)

        # With ignore_missing=False (default), result should be NaN
        # because there's a missing value
        assert numpy.isnan(result["mean"])
        assert numpy.isnan(result["stddev"])
        assert result["valid_count"] == 4  # valid_count is always computed

    def test_missing_values_with_validity_tuple(self):
        """Should accept (values, validity) tuple format."""
        values = [1.0, 2.0, 99.0, 4.0, 5.0]
        validity = [True, True, False, True, True]
        f = xfuncs.xfunc_numeric_summary((values, validity))

        cube = xcube([])
        result = f.calculate(cube)

        assert numpy.isnan(result["mean"])
        assert result["valid_count"] == 4

    def test_ignore_missing_true(self):
        """With ignore_missing=True, missing values should be skipped."""
        arr = [1.0, 2.0, float("nan"), 4.0, 5.0]
        f = xfuncs.xfunc_numeric_summary(arr, ignore_missing=True)

        cube = xcube([])
        result = f.calculate(cube)

        # Mean of [1, 2, 4, 5] = 3.0
        assert result["mean"] == pytest.approx(3.0)
        assert result["valid_count"] == 4
        assert result["sum"] == pytest.approx(12.0)


class TestNumericSummaryWithWeights:
    """Test weighted statistics."""

    def test_weighted_mean(self):
        """Weighted mean should be computed correctly."""
        arr = [1.0, 2.0, 3.0, 4.0, 5.0]
        weights = [1.0, 1.0, 1.0, 1.0, 6.0]  # Heavy weight on 5
        f = xfuncs.xfunc_numeric_summary(arr, weights=weights)

        cube = xcube([])
        result = f.calculate(cube)

        # Weighted mean = (1*1 + 2*1 + 3*1 + 4*1 + 5*6) / (1+1+1+1+6) = 40/10 = 4.0
        assert result["mean"] == pytest.approx(4.0)

    def test_weighted_quantiles(self):
        """Weighted quantiles should be computed correctly."""
        arr = [1.0, 2.0, 3.0, 4.0, 5.0]
        weights = [1.0, 1.0, 1.0, 1.0, 6.0]  # Heavy weight on 5
        f = xfuncs.xfunc_numeric_summary(
            arr,
            weights=weights,
            include={"quantiles": [0.5]}
        )

        cube = xcube([])
        result = f.calculate(cube)

        # With heavy weight on 5, median should be higher than 3
        assert result["quantiles"][0.5] > 3.0

    def test_valid_count_is_unweighted(self):
        """valid_count should always be unweighted (count of non-missing rows)."""
        arr = [1.0, 2.0, 3.0, 4.0, 5.0]
        weights = [10.0, 10.0, 10.0, 10.0, 10.0]
        f = xfuncs.xfunc_numeric_summary(arr, weights=weights)

        cube = xcube([])
        result = f.calculate(cube)

        # valid_count should be 5, not 50
        assert result["valid_count"] == 5


class TestNumericSummaryWithDimensions:
    """Test with cube dimensions (binned results)."""

    arr1 = [1, 0, 1, 0, 0]  # iindex for dimension
    fact = [10.0, 20.0, 30.0, 40.0, 50.0]

    def test_1d_cube(self):
        """Should compute statistics per bin in 1D cube."""
        f = xfuncs.xfunc_numeric_summary(self.fact, ignore_missing=True)

        cube = xcube([self.arr1])
        result = f.calculate(cube)

        # Bin 0: rows [1, 3, 4] -> values [20, 40, 50]
        # Bin 1: rows [0, 2] -> values [10, 30]
        assert result["mean"][0] == pytest.approx(numpy.mean([20, 40, 50]))
        assert result["mean"][1] == pytest.approx(numpy.mean([10, 30]))

        assert result["valid_count"][0] == 3
        assert result["valid_count"][1] == 2

        assert result["sum"][0] == pytest.approx(110.0)
        assert result["sum"][1] == pytest.approx(40.0)

    def test_2d_cube(self):
        """Should compute statistics per cell in 2D cube."""
        arr2 = [1, 0, 0, 1, 0]
        f = xfuncs.xfunc_numeric_summary(self.fact, ignore_missing=True)

        cube = xcube([self.arr1, arr2])
        result = f.calculate(cube)

        # Cell [0,0]: rows where arr1=0 AND arr2=0 -> rows [1, 4] -> values [20, 50]
        # Cell [0,1]: rows where arr1=0 AND arr2=1 -> rows [3] -> values [40]
        # Cell [1,0]: rows where arr1=1 AND arr2=0 -> rows [2] -> values [30]
        # Cell [1,1]: rows where arr1=1 AND arr2=1 -> rows [0] -> values [10]

        assert result["mean"][0, 0] == pytest.approx(35.0)  # mean([20, 50])
        assert result["mean"][0, 1] == pytest.approx(40.0)  # mean([40])
        assert result["mean"][1, 0] == pytest.approx(30.0)  # mean([30])
        assert result["mean"][1, 1] == pytest.approx(10.0)  # mean([10])


class TestNumericSummaryReturnFormat:
    """Test different return formats."""

    def test_return_missing_as_nan(self):
        """Default: missing cells should have NaN values."""
        arr = [1.0, 2.0, float("nan")]
        f = xfuncs.xfunc_numeric_summary(arr)

        cube = xcube([])
        result = f.calculate(cube)

        assert numpy.isnan(result["mean"])

    def test_return_missing_as_tuple(self):
        """With return_missing_as tuple, should return validity array."""
        arr = [1.0, 2.0, float("nan")]
        f = xfuncs.xfunc_numeric_summary(arr, return_missing_as=(0, False))

        cube = xcube([])
        result = f.calculate(cube)

        # Each statistic should have (value, validity) tuple
        mean_val, mean_valid = result["mean"]
        assert mean_val == 0  # sentinel value
        assert mean_valid == False  # not valid


class TestNumericSummaryEdgeCases:
    """Test edge cases."""

    def test_empty_after_filtering(self):
        """All values missing should return NaN for all statistics."""
        arr = [float("nan"), float("nan"), float("nan")]
        f = xfuncs.xfunc_numeric_summary(arr, ignore_missing=True)

        cube = xcube([])
        result = f.calculate(cube)

        assert numpy.isnan(result["mean"])
        assert numpy.isnan(result["stddev"])
        assert numpy.isnan(result["sum"])
        assert result["valid_count"] == 0

    def test_single_value(self):
        """Single value should work (stddev will be NaN due to ddof=1)."""
        arr = [5.0]
        f = xfuncs.xfunc_numeric_summary(arr)

        cube = xcube([])
        result = f.calculate(cube)

        assert result["mean"] == pytest.approx(5.0)
        assert result["sum"] == pytest.approx(5.0)
        assert result["valid_count"] == 1
        # stddev with ddof=1 and single value is NaN
        assert numpy.isnan(result["stddev"])

    def test_two_values_for_stddev(self):
        """Two values should give valid stddev."""
        arr = [0.0, 10.0]
        f = xfuncs.xfunc_numeric_summary(arr)

        cube = xcube([])
        result = f.calculate(cube)

        assert result["stddev"] == pytest.approx(numpy.std(arr, ddof=1))


class TestNumericSummaryMatchesIndividualXfuncs:
    """Verify numeric_summary matches individual xfunc results."""

    arr = [1.0, 2.0, float("nan"), 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    weights = [0.5, 1.0, 1.0, 1.5, 2.0, 0.25, 0.3, 0.99, 1.0, 0.8]
    dim1 = [0, 0, 1, 1, 0, 0, 1, 1, 0, 1]

    def test_matches_xfunc_mean(self):
        """numeric_summary mean should match xfunc_mean."""
        f_summary = xfuncs.xfunc_numeric_summary(
            self.arr,
            weights=self.weights,
            ignore_missing=True,
            include={"mean": True}
        )
        f_mean = xfuncs.xfunc_mean(self.arr, weights=self.weights, ignore_missing=True)

        cube = xcube([self.dim1])
        
        summary_result = f_summary.calculate(cube)
        # Use xcube.calculate for proper coordinate handling
        mean_result = cube.calculate([f_mean])[0]

        assert arr_eq(summary_result["mean"], mean_result)

    def test_matches_xfunc_stddev(self):
        """numeric_summary stddev should match xfunc_stddev."""
        f_summary = xfuncs.xfunc_numeric_summary(
            self.arr,
            weights=self.weights,
            ignore_missing=True,
            include={"stddev": True}
        )
        f_stddev = xfuncs.xfunc_stddev(
            self.arr, weights=self.weights, ignore_missing=True
        )

        cube = xcube([self.dim1])
        
        summary_result = f_summary.calculate(cube)
        # Use xcube.calculate for proper coordinate handling
        stddev_result = cube.calculate([f_stddev])[0]

        assert arr_eq(summary_result["stddev"], stddev_result)

    def test_matches_xfunc_valid_count(self):
        """numeric_summary valid_count should match xfunc_valid_count."""
        f_summary = xfuncs.xfunc_numeric_summary(
            self.arr,
            ignore_missing=True,
            include={"valid_count": True}
        )
        f_valid_count = xfuncs.xfunc_valid_count(self.arr, ignore_missing=True)

        cube = xcube([self.dim1])
        
        summary_result = f_summary.calculate(cube)
        # Use xcube.calculate for proper coordinate handling
        valid_count_result = cube.calculate([f_valid_count])[0]

        assert arr_eq(summary_result["valid_count"], valid_count_result)

    def test_matches_xfunc_quantile(self):
        """numeric_summary quantiles should match xfunc_quantile."""
        probs = [0.0, 0.25, 0.5, 0.75, 1.0]
        
        f_summary = xfuncs.xfunc_numeric_summary(
            self.arr,
            weights=self.weights,
            ignore_missing=True,
            include={"quantiles": probs}
        )

        cube = xcube([self.dim1])
        summary_result = f_summary.calculate(cube)

        for p in probs:
            f_quantile = xfuncs.xfunc_quantile(
                self.arr, p, weights=self.weights, ignore_missing=True
            )
            # Use xcube.calculate for proper coordinate handling
            quantile_result = cube.calculate([f_quantile])[0]
            
            assert arr_eq(
                summary_result["quantiles"][p], 
                quantile_result
            ), f"Mismatch at quantile {p}"
