"""Benchmarks comparing numeric summary computation approaches.

Compares:
1. Separate xfuncs: mean + stddev + valid_count + 5 quantiles (current approach)
2. Combined xfunc_numeric_summary (new approach)
"""

import numpy
from catii import xcubes, xfuncs

from . import UnitBenchmark


class TestNumericSummaryBenchmarks(UnitBenchmark):
    """Compare separate xfuncs vs combined numeric_summary."""

    def setUp(self):
        super().setUp()
        # Use realistic sizes for variable summary
        self.params.rows = 1_000_000

    def _make_numeric_with_missing(self, missing_rate=0.1):
        """Create numeric array with some missing values."""
        arr = numpy.random.randn(self.params.rows).astype(numpy.float64)
        # Add some missing values
        missing_mask = numpy.random.random(self.params.rows) < missing_rate
        arr[missing_mask] = numpy.nan
        return arr

    def _make_weights(self):
        """Create weight array."""
        return numpy.random.uniform(0.5, 2.0, self.params.rows).astype(numpy.float64)

    def test_separate_xfuncs_no_dims(self):
        """Benchmark: separate xfuncs for mean, stddev, valid_count, quantiles (no dimensions).

        This matches the actual get_summary pattern:
        - Call 1: mean, stddev, valid_count (batched)
        - Call 2: 5 quantiles (batched)
        """
        arr = self._make_numeric_with_missing()
        cube = xcubes.xcube([])  # No dimensions, scalar output

        with self.bench("separate_xfuncs.no_dims", threshold_ms=500):
            # Call 1: mean, stddev, valid_count (batched)
            f_mean = xfuncs.xfunc_mean(arr, ignore_missing=True)
            f_stddev = xfuncs.xfunc_stddev(arr, ignore_missing=True)
            f_valid_count = xfuncs.xfunc_valid_count(arr)
            cube.calculate([f_mean, f_stddev, f_valid_count])

            # Call 2: 5 quantiles (batched)
            quantile_funcs = [
                xfuncs.xfunc_quantile(arr, p, ignore_missing=True)
                for p in [0.0, 0.25, 0.5, 0.75, 1.0]
            ]
            cube.calculate(quantile_funcs)

    def test_numeric_summary_no_dims(self):
        """Benchmark: combined numeric_summary xfunc (no dimensions)."""
        arr = self._make_numeric_with_missing()
        cube = xcubes.xcube([])

        with self.bench("numeric_summary.no_dims", threshold_ms=200):
            f = xfuncs.xfunc_numeric_summary(
                arr,
                ignore_missing=True,
                include={
                    "mean": True,
                    "stddev": True,
                    "valid_count": True,
                    "quantiles": [0.0, 0.25, 0.5, 0.75, 1.0],
                },
            )
            result = f.calculate(cube)

    def test_separate_xfuncs_no_dims_weighted(self):
        """Benchmark: separate xfuncs with weights (no dimensions)."""
        arr = self._make_numeric_with_missing()
        weights = self._make_weights()
        cube = xcubes.xcube([])

        with self.bench("separate_xfuncs.no_dims.weighted", threshold_ms=1200):
            # Call 1: mean, stddev, valid_count
            f_mean = xfuncs.xfunc_mean(arr, weights=weights, ignore_missing=True)
            f_stddev = xfuncs.xfunc_stddev(arr, weights=weights, ignore_missing=True)
            f_valid_count = xfuncs.xfunc_valid_count(arr)
            cube.calculate([f_mean, f_stddev, f_valid_count])

            # Call 2: 5 quantiles (batched)
            quantile_funcs = [
                xfuncs.xfunc_quantile(arr, p, weights=weights, ignore_missing=True)
                for p in [0.0, 0.25, 0.5, 0.75, 1.0]
            ]
            cube.calculate(quantile_funcs)

    def test_numeric_summary_no_dims_weighted(self):
        """Benchmark: combined numeric_summary with weights (no dimensions)."""
        arr = self._make_numeric_with_missing()
        weights = self._make_weights()
        cube = xcubes.xcube([])

        with self.bench("numeric_summary.no_dims.weighted", threshold_ms=250):
            f = xfuncs.xfunc_numeric_summary(
                arr,
                weights=weights,
                ignore_missing=True,
                include={
                    "mean": True,
                    "stddev": True,
                    "valid_count": True,
                    "quantiles": [0.0, 0.25, 0.5, 0.75, 1.0],
                },
            )
            result = f.calculate(cube)

    def test_separate_xfuncs_1d(self):
        """Benchmark: separate xfuncs with 1 categorical dimension."""
        arr = self._make_numeric_with_missing()
        cat = self.categorical(indexed=False)
        cube = xcubes.xcube([cat])

        with self.bench("separate_xfuncs.1d", threshold_ms=1000):
            # Call 1: mean, stddev, valid_count
            f_mean = xfuncs.xfunc_mean(arr, ignore_missing=True)
            f_stddev = xfuncs.xfunc_stddev(arr, ignore_missing=True)
            f_valid_count = xfuncs.xfunc_valid_count(arr)
            cube.calculate([f_mean, f_stddev, f_valid_count])

            # Call 2: 5 quantiles (batched)
            quantile_funcs = [
                xfuncs.xfunc_quantile(arr, p, ignore_missing=True)
                for p in [0.0, 0.25, 0.5, 0.75, 1.0]
            ]
            cube.calculate(quantile_funcs)

    def test_numeric_summary_1d(self):
        """Benchmark: combined numeric_summary with 1 categorical dimension."""
        arr = self._make_numeric_with_missing()
        cat = self.categorical(indexed=False)
        cube = xcubes.xcube([cat])

        with self.bench("numeric_summary.1d", threshold_ms=400):
            f = xfuncs.xfunc_numeric_summary(
                arr,
                ignore_missing=True,
                include={
                    "mean": True,
                    "stddev": True,
                    "valid_count": True,
                    "quantiles": [0.0, 0.25, 0.5, 0.75, 1.0],
                },
            )
            result = f.calculate(cube)


class TestNumericSummarySeparateCalls(UnitBenchmark):
    """Compare single cube call with all measures vs separate calls.
    
    This demonstrates the difference between:
    1. Multiple xfuncs in one cube.calculate() call
    2. xfunc_numeric_summary: Single xfunc that shares intermediate computations
    """

    def setUp(self):
        super().setUp()
        self.params.rows = 1_000_000

    def _make_numeric_with_missing(self, missing_rate=0.1):
        arr = numpy.random.randn(self.params.rows).astype(numpy.float64)
        missing_mask = numpy.random.random(self.params.rows) < missing_rate
        arr[missing_mask] = numpy.nan
        return arr

    def test_separate_cube_calls(self):
        """Benchmark: 2 separate cube.calculate() calls (current get_summary pattern)."""
        arr = self._make_numeric_with_missing()
        cube = xcubes.xcube([])

        with self.bench("separate_cube_calls", threshold_ms=500):
            # Call 1: mean, stddev, valid_count
            f_mean = xfuncs.xfunc_mean(arr, ignore_missing=True)
            f_stddev = xfuncs.xfunc_stddev(arr, ignore_missing=True)
            f_valid_count = xfuncs.xfunc_valid_count(arr)
            cube.calculate([f_mean, f_stddev, f_valid_count])

            # Call 2: quantiles (batched in one call)
            quantile_funcs = [
                xfuncs.xfunc_quantile(arr, p, ignore_missing=True)
                for p in [0.0, 0.25, 0.5, 0.75, 1.0]
            ]
            cube.calculate(quantile_funcs)

    def test_all_xfuncs_in_single_call(self):
        """Benchmark: all xfuncs in ONE cube.calculate() call.
        
        Batch multiple independent xfuncs into
        a single cube call. Each xfunc still computes independently.
        """
        arr = self._make_numeric_with_missing()
        cube = xcubes.xcube([])

        with self.bench("all_xfuncs_in_single_call", threshold_ms=400):
            # All xfuncs created independently
            f_mean = xfuncs.xfunc_mean(arr, ignore_missing=True)
            f_stddev = xfuncs.xfunc_stddev(arr, ignore_missing=True)
            f_valid_count = xfuncs.xfunc_valid_count(arr)
            quantile_funcs = [
                xfuncs.xfunc_quantile(arr, p, ignore_missing=True)
                for p in [0.0, 0.25, 0.5, 0.75, 1.0]
            ]
            # All in one cube.calculate() call
            all_funcs = [f_mean, f_stddev, f_valid_count] + quantile_funcs
            cube.calculate(all_funcs)

    def test_numeric_summary_single_xfunc(self):
        """Benchmark: single xfunc_numeric_summary (shares intermediate computations).
        
        This is fundamentally different: ONE xfunc that internally shares:
        - sum/count for mean calculation
        - mean for stddev calculation  
        - sorted array for all quantiles
        """
        arr = self._make_numeric_with_missing()
        cube = xcubes.xcube([])

        with self.bench("numeric_summary_xfunc", threshold_ms=200):
            f = xfuncs.xfunc_numeric_summary(
                arr,
                ignore_missing=True,
                include={
                    "mean": True,
                    "stddev": True,
                    "valid_count": True,
                    "quantiles": [0.0, 0.25, 0.5, 0.75, 1.0],
                },
            )
            result = f.calculate(cube)


class TestExplainTheDifference(UnitBenchmark):
    """Benchmarks that isolate WHY numeric_summary is faster."""

    def setUp(self):
        super().setUp()
        self.params.rows = 1_000_000

    def _make_numeric_with_missing(self, missing_rate=0.1):
        arr = numpy.random.randn(self.params.rows).astype(numpy.float64)
        missing_mask = numpy.random.random(self.params.rows) < missing_rate
        arr[missing_mask] = numpy.nan
        return arr

    def test_quantiles_separate(self):
        """Each quantile xfunc sorts the array independently."""
        arr = self._make_numeric_with_missing()
        cube = xcubes.xcube([])

        with self.bench("quantiles_separate_5x_sort", threshold_ms=300):
            # 5 separate xfuncs = 5 separate sorts!
            quantile_funcs = [
                xfuncs.xfunc_quantile(arr, p, ignore_missing=True)
                for p in [0.0, 0.25, 0.5, 0.75, 1.0]
            ]
            cube.calculate(quantile_funcs)

    def test_quantiles_via_numeric_summary(self):
        """numeric_summary sorts once, extracts all 5 quantiles."""
        arr = self._make_numeric_with_missing()
        cube = xcubes.xcube([])

        with self.bench("quantiles_via_summary_1x_sort", threshold_ms=100):
            # 1 xfunc = 1 sort, then 5 lookups
            f = xfuncs.xfunc_numeric_summary(
                arr,
                ignore_missing=True,
                include={"quantiles": [0.0, 0.25, 0.5, 0.75, 1.0]},
            )
            result = f.calculate(cube)

    def test_mean_and_stddev_separate(self):
        """Separate xfuncs: mean computes sum/count, stddev recomputes them."""
        arr = self._make_numeric_with_missing()
        cube = xcubes.xcube([])

        with self.bench("mean_stddev_separate", threshold_ms=50):
            f_mean = xfuncs.xfunc_mean(arr, ignore_missing=True)
            f_stddev = xfuncs.xfunc_stddev(arr, ignore_missing=True)
            cube.calculate([f_mean, f_stddev])

    def test_mean_and_stddev_via_numeric_summary(self):
        """numeric_summary: computes sum/count once, reuses for both mean and stddev."""
        arr = self._make_numeric_with_missing()
        cube = xcubes.xcube([])

        with self.bench("mean_stddev_via_summary", threshold_ms=50):
            f = xfuncs.xfunc_numeric_summary(
                arr,
                ignore_missing=True,
                include={"mean": True, "stddev": True},
            )
            result = f.calculate(cube)
