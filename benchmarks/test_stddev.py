"""Benchmarks for xfunc_stddev performance.

These benchmarks test stddev performance across different configurations:
- 1D dimension with 1D numeric fact (baseline)
- 2D dimension (250 categories x 3 axes) with 1D numeric fact
- 1D dimension with 2D numeric fact (19 subvariables)

All configurations are tested both unweighted and weighted.
"""

import numpy
import pytest

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


class TestStddevManySubvarsFewCategories(UnitBenchmark):
    """Benchmark stddev with 2D dimension having many subvariables.

    This reproduces the production scenario where:
    - Dimension: 2D categorical array with 200 subvariables, each with 3 categories
    - Fact: 2D numeric array with matching 200 subvariables
    - 500K rows

    The cube scaffold iterates over the 200 dimension subvariables. For each,
    it computes stddev of the 1D fact slice grouped by the 3 categories.
    
    The bottleneck is 200 separate stddev calculations, each on 500K rows × 3 cats.
    """

    def setUp(self):
        super().setUp()
        self.params.rows = 50000
        self.params.categories = 3  # 3 categories per subvar
        self.params.density = 1.0  # Full density like production
        # Note: axes is set per-test since we vary dimension subvars

    def numeric_2d(self, subvars):
        """Return a 2D numeric array with the given number of subvariables."""
        base = self.numeric()
        return numpy.column_stack([base + i for i in range(subvars)])

    @pytest.mark.skip(reason="Slow to run, keep to document a slow scenario")
    def test_stddev_2d_dim_200subvars_3cats(self):
        """Stddev with 2D dim (3 cats × 200 subvars): the timeout case.

        This is the configuration that timed out at 120s in production.
        The scaffold iterates 200 times, each computing stddev over 3 categories.
        """
        self.params.axes = (200,)  # 200 dimension subvariables
        CA = self.categorical(indexed=False, array=True)  # 3 cats x 269 axes
        N = self.numeric_2d(200)
        self.params["dim_subvars"] = 200
        cube = xcubes.xcube([CA])
        # Current implementation times out at 120s.
        with self.bench("xcube.stddev.2d_dim_3x200.fact_200", threshold_ms=120000):
            cube.calculate([xfuncs.xfunc_stddev(N)])

    def test_stddev_2d_dim_23subvars_3cats(self):
        """Stddev with 2D dim (3 cats × 20 subvars): the slow but working case.

        The scaffold iterates 20 times, each computing stddev over 3 categories.
        """
        self.params.axes = (20,)  # 20 dimension subvariables
        CA = self.categorical(indexed=False, array=True)  # 3 cats x 20 axes
        N = self.numeric_2d(20)
        self.params["dim_subvars"] = 20
        cube = xcubes.xcube([CA])
        with self.bench("xcube.stddev.2d_dim_3x20.fact_20", threshold_ms=20000):
            cube.calculate([xfuncs.xfunc_stddev(N)])


class TestStddevProfile(UnitBenchmark):
    """Profile stddev with 1 dimension subvar to understand per-iteration cost."""

    def setUp(self):
        super().setUp()
        self.params.rows = 561000
        self.params.categories = 3
        self.params.density = 1.0

    def numeric_2d(self, subvars):
        base = self.numeric()
        return numpy.column_stack([base + i for i in range(subvars)])

    def test_stddev_1_dim_subvar_269_fact_cols(self):
        """Single dimension subvar with 269-column fact array.
        
        This isolates the cost of one stddev calculation on a 269-column fact.
        If this is ~1s, then 269 iterations = 269s (explaining the timeout).
        """
        self.params.axes = (1,)  # 1 dimension subvariable
        CA = self.categorical(indexed=False, array=True)  # 3 cats x 1 axis
        N = self.numeric_2d(269)  # 269-column fact
        self.params["dim_subvars"] = 1
        self.params["fact_cols"] = 269
        cube = xcubes.xcube([CA])
        with self.bench("xcube.stddev.1_dim_subvar.269_fact_cols", threshold_ms=5000):
            cube.calculate([xfuncs.xfunc_stddev(N)])

    def test_stddev_1_dim_subvar_1_fact_col(self):
        """Single dimension subvar with 1-column fact array.
        
        This isolates the per-column cost.
        """
        self.params.axes = (1,)  # 1 dimension subvariable
        CA = self.categorical(indexed=False, array=True)  # 3 cats x 1 axis
        N = self.numeric()  # 1-column fact
        self.params["dim_subvars"] = 1
        self.params["fact_cols"] = 1
        cube = xcubes.xcube([CA])
        with self.bench("xcube.stddev.1_dim_subvar.1_fact_col", threshold_ms=500):
            cube.calculate([xfuncs.xfunc_stddev(N)])

    def test_stddev_269_scaffolds_with_sliced_facts(self):
        """269 scaffold iterations, each with matching 1-col fact slice.
        
        This demonstrates what performance SHOULD be if the fact array were
        sliced to match each scaffold iteration. Instead of processing all 269
        columns 269 times (269×269=72K operations), this processes 1 column 
        269 times (269 operations).
        
        Expected: ~5s (much faster than 269 × 8.2s = 2200s with full fact)
        """
        self.params.axes = (269,)  # 269 dimension subvariables  
        CA = self.categorical(indexed=False, array=True)  # 3 cats x 269 axes
        self.params["dim_subvars"] = 269
        self.params["approach"] = "sliced_1col_per_scaffold"
        
        # Create the data once
        base = self.numeric()
        
        def compute_sliced():
            # Manually iterate like scaffold does, but with sliced facts
            for i in range(269):
                # Slice dimension to single scaffold position
                CA_slice = CA[:, i]  # shape (rows,) - single column
                fact_slice = base + i  # 1D fact for this scaffold position
                mini_cube = xcubes.xcube([CA_slice], interacting_shape=(3,))
                mini_cube.calculate([xfuncs.xfunc_stddev(fact_slice)])
        
        with self.bench("xcube.stddev.269_scaffolds_sliced_facts", threshold_ms=15000):
            compute_sliced()

    def test_stddev_2d_fact_aligned_with_scaffold(self):
        """Test stddev with 2D fact whose columns match scaffold subvars.
        
        This is the actual production case: 
        - Dimension: 2D categorical with 23 subvars
        - Fact: 2D numeric with 23 columns
        - Each scaffold iteration should only process the corresponding fact column
        
        With the fix, this should be ~23x faster than processing all columns.
        Current (broken): 23 iterations × 23 columns = 529 column-iterations
        Fixed: 23 iterations × 1 column = 23 column-iterations
        """
        self.params.axes = (23,)  # 23 dimension subvariables
        CA = self.categorical(indexed=False, array=True)  # 3 cats x 23 axes
        N = self.numeric_2d(23)  # 23-column fact, aligned with scaffold
        self.params["dim_subvars"] = 23
        self.params["fact_cols"] = 23
        self.params["approach"] = "aligned_fact_columns"
        
        cube = xcubes.xcube([CA])
        
        # This should be fast (~500ms) if fact is sliced per scaffold iteration
        # But is slow (~11s) if all 23 columns are processed every iteration
        with self.bench("xcube.stddev.aligned_2d_fact", threshold_ms=2000):
            cube.calculate([xfuncs.xfunc_stddev(N)])

    def test_stddev_2d_fact_aligned_269_subvars(self):
        """Test stddev with 269 aligned subvars (production timeout case).
        
        This reproduces the production timeout:
        - Dimension: 2D categorical with 269 subvars
        - Fact: 2D numeric with 269 columns
        
        Without fix: 269 × 269 = 72,361 column-iterations = timeout
        With fix: 269 × 1 = 269 column-iterations = ~6s
        """
        self.params.axes = (269,)  # 269 dimension subvariables
        CA = self.categorical(indexed=False, array=True)  # 3 cats x 269 axes
        N = self.numeric_2d(269)  # 269-column fact, aligned with scaffold
        self.params["dim_subvars"] = 269
        self.params["fact_cols"] = 269
        self.params["approach"] = "aligned_fact_columns"
        
        cube = xcubes.xcube([CA])
        
        # With the fix, this should complete in ~6-7s
        with self.bench("xcube.stddev.aligned_269_subvars", threshold_ms=15000):
            cube.calculate([xfuncs.xfunc_stddev(N)])

