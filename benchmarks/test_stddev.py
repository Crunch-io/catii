"""Benchmarks for xfunc_stddev performance.

These benchmarks test stddev performance across different configurations:
- 1D dimension with 1D numeric fact (baseline)
- 2D dimension (250 categories x 3 axes) with 1D numeric fact
- 1D dimension with 2D numeric fact (19 subvariables)

All configurations are tested both unweighted and weighted.
"""

import numpy

from catii import xcubes, xfuncs

from . import UnitBenchmark


class TestStddev1D(UnitBenchmark):
    """Benchmark stddev with 1D dimension and 1D numeric fact."""

    def test_stddev_1d_dim_1d_fact(self):
        """Stddev with 1D dimension, 1D fact, unweighted."""
        C = self.categorical(indexed=False)
        N = self.numeric()
        cube = xcubes.xcube([C])
        with self.bench("xcube.stddev.1d_dim.1d_fact", threshold_ms=100):
            cube.calculate([xfuncs.xfunc_stddev(N)])

    def test_stddev_1d_dim_1d_fact_weighted(self):
        """Stddev with 1D dimension, 1D fact, weighted."""
        C = self.categorical(indexed=False)
        N = self.numeric()
        W = self.numeric()
        cube = xcubes.xcube([C])
        with self.bench("xcube.stddev.1d_dim.1d_fact.wt", threshold_ms=100):
            cube.calculate([xfuncs.xfunc_stddev(N, weights=W)])


class TestStddev2DDim(UnitBenchmark):
    """Benchmark stddev with 2D dimension (250 categories x 3 axes)."""

    def setUp(self):
        super().setUp()
        # 250 categories, 3 axes (like genre with like/dislike responses)
        self.params.categories = 250
        self.params.axes = (3,)

    def test_stddev_2d_dim_1d_fact(self):
        """Stddev with 2D dimension (250x3), 1D fact, unweighted."""
        CA = self.categorical(indexed=False, array=True)
        N = self.numeric()
        cube = xcubes.xcube([CA])
        with self.bench("xcube.stddev.2d_dim_250x3.1d_fact", threshold_ms=200):
            cube.calculate([xfuncs.xfunc_stddev(N)])

    def test_stddev_2d_dim_1d_fact_weighted(self):
        """Stddev with 2D dimension (250x3), 1D fact, weighted."""
        CA = self.categorical(indexed=False, array=True)
        N = self.numeric()
        W = self.numeric()
        cube = xcubes.xcube([CA])
        with self.bench("xcube.stddev.2d_dim_250x3.1d_fact.wt", threshold_ms=200):
            cube.calculate([xfuncs.xfunc_stddev(N, weights=W)])


class TestStddev2DDimCrossed(UnitBenchmark):
    """Benchmark stddev with 2D dimension (250x3) crossed with 1D dimension.

    This tests the pattern where a dimension with subvariables is crossed
    with another dimension, producing a cube with shape (3, 250, 250) when
    the 2D dim has 250 categories and 3 subvariables, crossed with a 1D dim
    of 250 categories.
    """

    def setUp(self):
        super().setUp()
        self.params.rows = 1000000
        self.params.categories = 250
        self.params.axes = (3,)

    def categorical_1d(self):
        """Return a 1D categorical array (no additional axes)."""
        # Temporarily override axes to get a 1D array
        old_axes = self.params.axes
        self.params.axes = ()
        try:
            return self.categorical(indexed=False, array=False)
        finally:
            self.params.axes = old_axes

    def test_stddev_2d_dim_x_1d_dim_1d_fact(self):
        """Stddev with 2D dim (250x3) x 1D dim, 1D fact, unweighted."""
        CA = self.categorical(indexed=False, array=True)  # 250 cats x 3 axes
        C = self.categorical_1d()  # 250 cats, 1D
        N = self.numeric()
        cube = xcubes.xcube([CA, C])
        with self.bench("xcube.stddev.2d_dim_250x3_x_1d_dim.1d_fact", threshold_ms=250):
            cube.calculate([xfuncs.xfunc_stddev(N)])

    def test_stddev_2d_dim_x_1d_dim_1d_fact_weighted(self):
        """Stddev with 2D dim (250x3) x 1D dim, 1D fact, weighted."""
        CA = self.categorical(indexed=False, array=True)  # 250 cats x 3 axes
        C = self.categorical_1d()  # 250 cats, 1D
        N = self.numeric()
        W = self.numeric()
        cube = xcubes.xcube([CA, C])
        with self.bench("xcube.stddev.2d_dim_250x3_x_1d_dim.1d_fact.wt", threshold_ms=250):
            cube.calculate([xfuncs.xfunc_stddev(N, weights=W)])


class TestStddev2DFact(UnitBenchmark):
    """Benchmark stddev with 2D numeric fact (19 subvariables)."""

    def numeric_2d(self, subvars=19):
        """Return a 2D numeric array with the given number of subvariables."""
        base = self.numeric()
        # Create 2D array by stacking shifted versions
        return numpy.column_stack([base + i for i in range(subvars)])

    def test_stddev_1d_dim_2d_fact_19subvars(self):
        """Stddev with 1D dimension, 2D fact (19 subvars), unweighted."""
        C = self.categorical(indexed=False)
        N = self.numeric_2d(19)
        self.params["subvars"] = 19
        cube = xcubes.xcube([C])
        with self.bench("xcube.stddev.1d_dim.2d_fact_19subvars", threshold_ms=1200):
            cube.calculate([xfuncs.xfunc_stddev(N)])

    def test_stddev_1d_dim_2d_fact_19subvars_weighted(self):
        """Stddev with 1D dimension, 2D fact (19 subvars), weighted."""
        C = self.categorical(indexed=False)
        N = self.numeric_2d(19)
        W = self.numeric()
        self.params["subvars"] = 19
        cube = xcubes.xcube([C])
        with self.bench("xcube.stddev.1d_dim.2d_fact_19subvars.wt", threshold_ms=1400):
            cube.calculate([xfuncs.xfunc_stddev(N, weights=W)])


class TestStddev2DDim2DFact(UnitBenchmark):
    """Benchmark stddev with 2D dimension AND 2D numeric fact.

    This tests the pattern where:
    - dimension: categorical variable with subvariables (2D: categories x subvars)
    - fact: numeric array with multiple subvariables (2D: rows x numeric subvars)

    The cube iterates over the dimension's subvariables (scaffold), and for each,
    computes stddev of the 2D numeric fact grouped by the categories.
    """

    def setUp(self):
        super().setUp()
        self.params.categories = 250
        self.params.axes = (3,)  # 3 subvariables in the dimension

    def numeric_2d(self, subvars=19):
        """Return a 2D numeric array with the given number of subvariables."""
        base = self.numeric()
        return numpy.column_stack([base + i for i in range(subvars)])

    def test_stddev_2d_dim_2d_fact(self):
        """Stddev with 2D dim (250x3), 2D fact (19 subvars), unweighted."""
        CA = self.categorical(indexed=False, array=True)  # 250 cats x 3 axes
        N = self.numeric_2d(19)
        self.params["fact_subvars"] = 19
        cube = xcubes.xcube([CA])
        with self.bench("xcube.stddev.2d_dim_250x3.2d_fact_19subvars", threshold_ms=4000):
            cube.calculate([xfuncs.xfunc_stddev(N)])

    def test_stddev_2d_dim_2d_fact_weighted(self):
        """Stddev with 2D dim (250x3), 2D fact (19 subvars), weighted."""
        CA = self.categorical(indexed=False, array=True)  # 250 cats x 3 axes
        N = self.numeric_2d(19)
        W = self.numeric()
        self.params["fact_subvars"] = 19
        cube = xcubes.xcube([CA])
        with self.bench("xcube.stddev.2d_dim_250x3.2d_fact_19subvars.wt", threshold_ms=4500):
            cube.calculate([xfuncs.xfunc_stddev(N, weights=W)])
