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


class TestStddevOptimization(UnitBenchmark):
    """Benchmark comparing old (column-loop) vs new (vectorized) stddev algorithm.

    The original xfunc_stddev implementation looped over columns when the fact array
    was 2D, calling _fill_one_by_coordinates for each column separately.
    This was O(columns * categories) with Python loop overhead per column.

    The optimized implementation loops over bins (categories) and uses vectorized
    numpy operations across all columns at once, which is much faster when the
    number of columns (subvariables) is large.

    This benchmark simulates the production case: 269 subvariables, 100K rows, 3 cats.
    """

    def setUp(self):
        super().setUp()
        self.params.rows = 100_000
        self.params.categories = 3

    def test_stddev_optimization_comparison(self):
        """Compare old column-loop vs new vectorized algorithm."""
        import time

        n_subvars = 269
        numpy.random.seed(42)

        dim = numpy.random.randint(
            0, self.params.categories, size=self.params.rows, dtype=numpy.uint8
        )
        fact = numpy.random.randn(self.params.rows, n_subvars)

        # Add some missing values (~5%)
        mask = numpy.random.random((self.params.rows, n_subvars)) < 0.05
        fact[mask] = float("nan")

        cube = xcubes.xcube([dim])

        # Simulate OLD algorithm: loop over columns inside fill
        def old_fill_algorithm():
            f = xfuncs.xfunc_stddev(fact, ignore_missing=True)
            regions = f.get_initial_regions(cube)
            regions = f.flat_regions(regions)
            coordinates = sum(cube.strided_dims())

            for col in range(f.summables.shape[1]):
                f._fill_one_by_coordinates(
                    [r[:, col] for r in regions],
                    coordinates,
                    f.summables[:, col],
                    f.wsummables[:, col],
                    f.countables[:, col],
                    f.validity[:, col],
                )
            return f.reduce(cube, regions)

        # NEW algorithm: vectorized fill
        def new_fill_algorithm():
            return cube.stddev(fact, ignore_missing=True)

        # Benchmark OLD
        start_old = time.perf_counter()
        old_result = old_fill_algorithm()
        elapsed_old = time.perf_counter() - start_old

        # Benchmark NEW
        start_new = time.perf_counter()
        new_result = new_fill_algorithm()
        elapsed_new = time.perf_counter() - start_new

        # Verify results match
        assert numpy.allclose(old_result, new_result, equal_nan=True), \
            "Old and new algorithm results differ!"

        speedup = elapsed_old / elapsed_new

        self.params["subvars"] = n_subvars
        self.params["old_ms"] = elapsed_old * 1000
        self.params["new_ms"] = elapsed_new * 1000
        self.params["speedup"] = f"{speedup:.1f}x"

        print(f"\nOptimization comparison ({self.params.rows:,} rows x {n_subvars} subvars x {self.params.categories} cats):")
        print(f"  Old (column-loop): {elapsed_old:.3f}s")
        print(f"  New (vectorized):  {elapsed_new:.3f}s")
        print(f"  Speedup: {speedup:.1f}x")

        # The new approach should be faster
        assert speedup > 1.2, f"Expected speedup > 1.2x, got {speedup:.1f}x"
