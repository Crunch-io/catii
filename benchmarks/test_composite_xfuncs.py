"""Benchmarks for CompositeXfunc shared intermediate computations.

These benchmarks compare the performance of running compatible xfuncs
(like mean and stddev) together via CompositeXfunc versus running them
separately without shared intermediates.
"""

from catii import xcubes, xfuncs

from . import UnitBenchmark


class Test_composite_xfunc_1d(UnitBenchmark):
    """Benchmark CompositeXfunc with 1D dimensions."""

    def test_xcube_1d_mean_stddev_composite(self):
        """Mean and stddev together (auto-grouped, shared intermediates)."""
        N = self.numeric()
        C = self.categorical(indexed=False)
        cube = xcubes.xcube([C])
        fmean = xfuncs.xfunc_mean(N)
        fstddev = xfuncs.xfunc_stddev(N)
        with self.bench("xcube.1d.mean_stddev.composite", 60):
            cube.calculate([fmean, fstddev])

    def test_xcube_1d_mean_stddev_separate(self):
        """Mean and stddev separately (no shared intermediates).
        
        Uses different arrays (N * 2) to prevent auto-grouping.
        """
        N = self.numeric()
        N2 = N * 2  # Different values to prevent grouping
        C = self.categorical(indexed=False)
        cube = xcubes.xcube([C])
        fmean = xfuncs.xfunc_mean(N)
        fstddev = xfuncs.xfunc_stddev(N2)
        with self.bench("xcube.1d.mean_stddev.separate", 80):
            cube.calculate([fmean, fstddev])


class Test_composite_xfunc_1d_x_1d(UnitBenchmark):
    """Benchmark CompositeXfunc with crossed 1D dimensions."""

    def test_xcube_1d_x_1d_mean_stddev_composite(self):
        """Mean and stddev together with crossed dims (shared intermediates)."""
        N = self.numeric()
        C = self.categorical(indexed=False)
        cube = xcubes.xcube([C, C])
        fmean = xfuncs.xfunc_mean(N)
        fstddev = xfuncs.xfunc_stddev(N)
        with self.bench("xcube.1d_x_1d.mean_stddev.composite", 60):
            cube.calculate([fmean, fstddev])

    def test_xcube_1d_x_1d_mean_stddev_separate(self):
        """Mean and stddev separately with crossed dims (no shared intermediates)."""
        N = self.numeric()
        N2 = N * 2
        C = self.categorical(indexed=False)
        cube = xcubes.xcube([C, C])
        fmean = xfuncs.xfunc_mean(N)
        fstddev = xfuncs.xfunc_stddev(N2)
        with self.bench("xcube.1d_x_1d.mean_stddev.separate", 80):
            cube.calculate([fmean, fstddev])


class Test_composite_xfunc_1d_wt(UnitBenchmark):
    """Benchmark CompositeXfunc with weighted xfuncs."""

    def test_xcube_1d_mean_stddev_wt_composite(self):
        """Weighted mean and stddev together (shared intermediates)."""
        N = self.numeric()
        W = self.numeric()
        C = self.categorical(indexed=False)
        cube = xcubes.xcube([C])
        fmean = xfuncs.xfunc_mean(N, weights=W)
        fstddev = xfuncs.xfunc_stddev(N, weights=W)
        with self.bench("xcube.1d.mean_stddev.wt.composite", 60):
            cube.calculate([fmean, fstddev])

    def test_xcube_1d_mean_stddev_wt_separate(self):
        """Weighted mean and stddev separately (no shared intermediates)."""
        N = self.numeric()
        N2 = N * 2
        W = self.numeric()
        C = self.categorical(indexed=False)
        cube = xcubes.xcube([C])
        fmean = xfuncs.xfunc_mean(N, weights=W)
        fstddev = xfuncs.xfunc_stddev(N2, weights=W)
        with self.bench("xcube.1d.mean_stddev.wt.separate", 80):
            cube.calculate([fmean, fstddev])
