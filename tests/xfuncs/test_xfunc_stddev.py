import operator
import sys
from functools import reduce

import numpy
import pytest

from catii import xcube, xfuncs

from .. import arr_eq

MAXFLOAT = sys.float_info.max
arr1 = [1, 0, 1, 0, 1, 0, 1, 0]  # iindex({(1,): [0, 2, 4, 6]}, 0, (8,))
arr2 = [1, 0, 0, 1, 0, 0, 1, 0]  # iindex({(1,): [0, 3, 6]}, 0, (8,))
wt = numpy.array([0.25, 0.3, 0.99, 1.0, float("nan"), 0.5, 0.75, 0.0])

# The cube of arr1 x arr2 has rowids:
#      ___arr 2___
#      0       1
# a|0  [1, 5, 7]  [3]
# r|
# r|
# 1|1  [2, 4]     [0, 6]
#
# and therefore weights:
#      ________arr 2_________
#      0                1
# a|0  [0.3, 0.5, 0.0]  [1.0]
# r|
# r|
# 1|1  [0.99, nan]      [0.25, 0.75]


class TestXfuncStddevWorkflow:
    def _test_unweighted_workflow(self, factvar):
        # Other tests often simply test output; here we test intermediate results.
        f = xfuncs.xfunc_stddev(factvar)

        cube = xcube([])
        stddevs, valids, missings = f.get_initial_regions(cube)
        assert arr_eq(stddevs, [float("nan")])
        assert valids.tolist() == [0]
        assert missings.tolist() == [0]
        coordinates = None
        f.fill(coordinates, (stddevs, valids, missings))
        assert stddevs.tolist() == [36.2832770085089]
        assert valids.tolist() == [7]
        assert missings.tolist() == [1]
        assert numpy.isnan(f.reduce(cube, (stddevs, valids, missings)))

        cube = xcube([arr1])
        stddevs, valids, missings = f.get_initial_regions(cube)
        assert arr_eq(stddevs, [float("nan"), float("nan")])
        assert valids.tolist() == [0, 0]
        assert missings.tolist() == [0, 0]
        coordinates = reduce(operator.add, cube.strided_dims())
        f.fill(coordinates, (stddevs, valids, missings))
        assert arr_eq(stddevs, [48.027769744874334, float("nan")])
        assert valids.tolist() == [4, 4]
        assert missings.tolist() == [0, 1]
        assert arr_eq(
            f.reduce(cube, (stddevs, valids, missings)),
            [48.027769744874334, float("nan")],
        )

        cube = xcube([arr1, arr2])
        stddevs, valids, missings = f.get_initial_regions(cube)
        assert arr_eq(
            stddevs, [[float("nan"), float("nan")], [float("nan"), float("nan")]]
        )
        assert valids.tolist() == [[0, 0], [0, 0]]
        assert missings.tolist() == [[0, 0], [0, 0]]
        coordinates = reduce(operator.add, cube.strided_dims())
        f.fill(coordinates, (stddevs, valids, missings))
        assert arr_eq(
            stddevs,
            [[55.46169849544819, float("nan")], [float("nan"), 4.242640687119285]],
        )
        assert valids.tolist() == [[3, 1], [2, 2]]
        assert missings.tolist() == [[0, 0], [1, 0]]
        assert arr_eq(
            f.reduce(cube, (stddevs, valids, missings)),
            [[55.46169849544819, float("nan")], [float("nan"), 4.242640687119285]],
        )

    def _test_weighted_workflow(self, factvar):
        # Other tests often simply test output; here we test intermediate results.
        f = xfuncs.xfunc_stddev(factvar, weights=wt)

        cube = xcube([arr1, arr2])
        stddevs, valids, missings = f.get_initial_regions(cube)
        assert arr_eq(
            stddevs, [[float("nan"), float("nan")], [float("nan"), float("nan")]]
        )
        assert valids.tolist() == [[0, 0], [0, 0]]
        assert missings.tolist() == [[0, 0], [0, 0]]
        coordinates = reduce(operator.add, cube.strided_dims())
        f.fill(coordinates, (stddevs, valids, missings))
        assert arr_eq(
            stddevs, [[2.37170825, float("nan")], [float("nan"), 3.67423461]],
        )
        assert valids.tolist() == [[3, 1], [2, 2]]
        assert missings.tolist() == [[0, 0], [2, 0]]
        assert arr_eq(
            f.reduce(cube, (stddevs, valids, missings)),
            [[2.37170825, float("nan")], [float("nan"), 3.67423461]],
        )

    def test_single_arr_with_nan_workflow(self):
        factvar = [1.0, 2.0, float("nan"), 4.0, 5.0, 6.0, 7.0, 100.0]
        self._test_unweighted_workflow(factvar)
        self._test_weighted_workflow(factvar)

    def test_tuple_factvar_workflow(self):
        factvar = (
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 100.0],
            [True, True, False, True, True, True, True, True],
        )
        self._test_unweighted_workflow(factvar)
        self._test_weighted_workflow(factvar)

    def test_tuple_factvar_with_nan_workflow(self):
        factvar = (
            [1.0, 2.0, float("nan"), 4.0, 5.0, 6.0, 7.0, 100.0],
            [True, True, False, True, True, True, True, True],
        )
        self._test_unweighted_workflow(factvar)
        self._test_weighted_workflow(factvar)


class TestXfuncMeanMissingness:
    # Make sure each combination of these options works properly:
    #  * ignore_missing = True/False
    #  * return_missing_as = NaN/(<sentinel value>, False)
    #  * at least one output cell which has no inputs
    #  * fact variable (with missings) as:
    #    * single array with NaN
    #    * no NaN, but separate "validity" array
    #    * includes NaN, but separate "validity" array
    #  * weights variable (with missings) as:
    #    * single array with NaN
    #    * no NaN, but separate "validity" array
    #    * includes NaN, but separate "validity" array
    arr_with_empty_cell = [0, 0, 0, 1, 0, 0, 0, 1]

    dirty_fact = [1.0, 2.0, float("nan"), 4.0, 5.0, 6.0, 7.0, 100.0]
    clean_fact = [1.0, 2.0, MAXFLOAT, 4.0, 5.0, 6.0, 7.0, 100.0]
    fact_validity = [True, True, False, True, True, True, True, True]
    fact_all_valid = [True] * 8

    dirty_weights = [9.0, 9.0, float("nan"), 9.0, 9.0, 9.0, 9.0, 9.0]
    clean_weights = [9.0, 9.0, MAXFLOAT, 9.0, 9.0, 9.0, 9.0, 9.0]
    weights_validity = [True, True, False, True, True, True, True, True]

    params = [
        [dirty_fact, None],
        [(clean_fact, fact_validity), None],
        [(dirty_fact, fact_validity), None],
        [dirty_fact, dirty_weights],
        [(clean_fact, fact_validity), dirty_weights],
        [(dirty_fact, fact_validity), dirty_weights],
        [(clean_fact, fact_all_valid), dirty_weights],
        [dirty_fact, (clean_weights, weights_validity)],
        [(clean_fact, fact_validity), (clean_weights, weights_validity)],
        [(dirty_fact, fact_validity), (clean_weights, weights_validity)],
        [(clean_fact, fact_all_valid), (clean_weights, weights_validity)],
        [dirty_fact, (dirty_weights, weights_validity)],
        [(clean_fact, fact_validity), (dirty_weights, weights_validity)],
        [(dirty_fact, fact_validity), (dirty_weights, weights_validity)],
        [(clean_fact, fact_all_valid), (dirty_weights, weights_validity)],
    ]

    @pytest.mark.parametrize("factvar,weights", params)
    def test_propagate_missing_return_nan(self, factvar, weights):
        # The cube of arr1 has rowids:
        # 0           1
        # [1, 3, 5, 7]  [0, 2, 4, 6]
        # and fact values:
        # 0           1
        # [2.0, 4.0, 6.0, 100.0]  [1.0, nan, 5.0, 7.0]
        stddevs = xcube([arr1]).stddev(factvar, weights)
        # Cell (1,) MUST be missing, because it had a missing input.
        assert arr_eq(stddevs, [48.02776974, float("nan")])

        # The cube of arr1 x arr_with_empty_cell has fact values:
        #      __________arr w__________
        #      0                    1
        # a|0  [2.0, 6.0, 100.0]    [4.0]
        # r|
        # r|
        # 1|1  [1.0, nan, 5.0, 7.0] []
        stddevs = xcube([arr1, self.arr_with_empty_cell]).stddev(factvar, weights)
        # Cell (1, 0) MUST be missing, because it had a missing input.
        # Cell (1, 1) MUST be missing, because it had no inputs.
        assert arr_eq(
            stddevs, [[2.82842712, 67.88225099], [float("nan"), float("nan")]]
        )

    @pytest.mark.parametrize("factvar,weights", params)
    def test_ignore_missing_return_nan(self, factvar, weights):
        # The cube of arr1 has rowids:
        # 0           1
        # [1, 3, 5, 7]  [0, 2, 4, 6]
        # and fact values:
        # 0           1
        # [2.0, 4.0, 6.0, 100.0]  [1.0, nan, 5.0, 7.0]
        stddevs = xcube([arr1]).stddev(factvar, weights, ignore_missing=True)
        # Cell (1,) MUST NOT be missing, because it had a valid input.
        assert arr_eq(stddevs, [48.02776974, 3.055050463303893])

        # The cube of arr1 x arr_with_empty_cell has fact values:
        #      __________arr w__________
        #      0                    1
        # a|0  [2.0, 6.0, 100.0]    [4.0]
        # r|
        # r|
        # 1|1  [1.0, nan, 5.0, 7.0] []
        stddevs = xcube([arr1, self.arr_with_empty_cell]).stddev(
            factvar, weights, ignore_missing=True
        )
        # Cell (1, 0) MUST NOT be missing, because it had a valid input.
        # Cell (1, 1) MUST be missing, because it had no inputs.
        assert arr_eq(
            stddevs, [[2.82842712, 67.88225099], [3.055050463303893, float("nan")]]
        )

    @pytest.mark.parametrize("factvar,weights", params)
    def test_propagate_missing_return_validity(self, factvar, weights):
        # The cube of arr1 has rowids:
        # 0           1
        # [1, 3, 5, 7]  [0, 2, 4, 6]
        # and fact values:
        # 0           1
        # [2.0, 4.0, 6.0, 100.0]  [1.0, nan, 5.0, 7.0]
        stddevs, validity = xcube([arr1]).stddev(
            factvar, weights, return_missing_as=(-1, False)
        )
        # Cell (1,) MUST be missing, because it had a missing input.
        assert arr_eq(stddevs, [48.02776974, -1])
        assert arr_eq(validity, [True, False])

        # The cube of arr1 x arr_with_empty_cell has fact values:
        #      __________arr w__________
        #      0                    1
        # a|0  [2.0, 6.0, 100.0]    [4.0]
        # r|
        # r|
        # 1|1  [1.0, nan, 5.0, 7.0] []
        stddevs, validity = xcube([arr1, self.arr_with_empty_cell]).stddev(
            factvar, weights, return_missing_as=(-1, False)
        )
        # Cell (1, 0) MUST be missing, because it had a missing input.
        # Cell (1, 1) MUST be missing, because it had no inputs.
        assert arr_eq(stddevs, [[2.82842712, 67.88225099], [-1, -1]])
        assert arr_eq(validity, [[True, True], [False, False]])

    @pytest.mark.parametrize("factvar,weights", params)
    def test_ignore_missing_return_validity(self, factvar, weights):
        # The cube of arr1 has rowids:
        # 0           1
        # [1, 3, 5, 7]  [0, 2, 4, 6]
        # and fact values:
        # 0           1
        # [2.0, 4.0, 6.0, 100.0]  [1.0, nan, 5.0, 7.0]
        stddevs, validity = xcube([arr1]).stddev(
            factvar, weights, ignore_missing=True, return_missing_as=(-1, False)
        )
        # Cell (1,) MUST NOT be missing, because it had a valid input.
        assert arr_eq(stddevs, [48.02776974, 3.055050463303893])
        assert arr_eq(validity, [True, True])

        # The cube of arr1 x arr_with_empty_cell has fact values:
        #      __________arr w__________
        #      0                    1
        # a|0  [2.0, 6.0, 100.0]    [4.0]
        # r|
        # r|
        # 1|1  [1.0, nan, 5.0, 7.0] []
        stddevs, validity = xcube([arr1, self.arr_with_empty_cell]).stddev(
            factvar, weights, ignore_missing=True, return_missing_as=(-1, False)
        )
        # Cell (1, 0) MUST NOT be missing, because it had a valid input.
        # Cell (1, 1) MUST be missing, because it had no inputs.
        assert arr_eq(stddevs, [[2.82842712, 67.88225099], [3.055050463303893, -1]])
        assert arr_eq(validity, [[True, True], [True, False]])


class TestXfuncStddev2DDim2DFact:
    """Test xfunc_stddev with 2D dimension AND 2D numeric fact array.

    This tests the pattern where:
    - dimension: categorical variable with subvariables (2D: rows x dim_subvars)
    - fact: numeric array with subvariables (2D: rows x fact_subvars)

    The cube iterates over the dimension's subvariables (scaffold), and for each,
    computes stddev of the 2D numeric fact grouped by the categories.
    """

    # 2D dimension array: 8 rows x 3 subvariables (like a categorical with 3 subvars)
    # Each column has 2 categories (0 and 1)
    dim_2d = numpy.array([
        [1, 0, 1],  # row 0
        [0, 1, 0],  # row 1
        [1, 0, 1],  # row 2
        [0, 1, 0],  # row 3
        [1, 0, 1],  # row 4
        [0, 1, 0],  # row 5
        [1, 0, 1],  # row 6
        [0, 1, 0],  # row 7
    ], dtype=numpy.uint8)

    # 2D numeric fact array: 8 rows x 2 subvariables
    fact_2d = numpy.array([
        [1.0, 10.0],   # row 0
        [2.0, 20.0],   # row 1
        [3.0, 30.0],   # row 2
        [4.0, 40.0],   # row 3
        [5.0, 50.0],   # row 4
        [6.0, 60.0],   # row 5
        [7.0, 70.0],   # row 6
        [8.0, 80.0],   # row 7
    ])

    # Weights array
    weights = numpy.array([0.5, 1.0, 1.5, 2.0, 0.5, 1.0, 1.5, 2.0])

    def test_2d_dim_2d_fact_unweighted(self):
        """Stddev with 2D dimension and 2D numeric fact, unweighted."""
        # dim_2d has shape (8, 3) -> 3 subvariables in the dimension
        # fact_2d has shape (8, 2) -> 2 subvariables in the fact
        # Output shape should be (3, 2, 2): (dim_subvars, categories, fact_subvars)
        stddevs = xcube([self.dim_2d]).stddev(self.fact_2d)

        # For dim subvar 0: cat=0 rows are [1,3,5,7], cat=1 rows are [0,2,4,6]
        #   fact col 0: cat=0 gets [2,4,6,8], cat=1 gets [1,3,5,7]
        #   fact col 1: cat=0 gets [20,40,60,80], cat=1 gets [10,30,50,70]
        assert stddevs.shape == (3, 2, 2)

        # Check dim subvar 0 results
        expected_subvar0_cat0_col0 = numpy.std([2, 4, 6, 8], ddof=1)
        expected_subvar0_cat0_col1 = numpy.std([20, 40, 60, 80], ddof=1)
        expected_subvar0_cat1_col0 = numpy.std([1, 3, 5, 7], ddof=1)
        expected_subvar0_cat1_col1 = numpy.std([10, 30, 50, 70], ddof=1)

        assert arr_eq(stddevs[0, 0, 0], expected_subvar0_cat0_col0)
        assert arr_eq(stddevs[0, 0, 1], expected_subvar0_cat0_col1)
        assert arr_eq(stddevs[0, 1, 0], expected_subvar0_cat1_col0)
        assert arr_eq(stddevs[0, 1, 1], expected_subvar0_cat1_col1)

    def test_2d_dim_2d_fact_weighted(self):
        """Stddev with 2D dimension and 2D numeric fact, weighted."""
        stddevs = xcube([self.dim_2d]).stddev(self.fact_2d, weights=self.weights)

        assert stddevs.shape == (3, 2, 2)
        # Verify we get numeric results (not NaN)
        assert not numpy.isnan(stddevs).any()

        # Verify weighted stddev is different from unweighted
        unweighted = xcube([self.dim_2d]).stddev(self.fact_2d)
        assert not arr_eq(stddevs, unweighted)

    def test_2d_dim_2d_fact_with_missing(self):
        """Stddev with 2D dim and 2D fact, with missing values in fact."""
        fact_with_nan = self.fact_2d.copy()
        fact_with_nan[2, 0] = float("nan")  # row 2, col 0 is missing

        stddevs = xcube([self.dim_2d]).stddev(fact_with_nan)
        assert stddevs.shape == (3, 2, 2)

        # Row 2 belongs to cat=1 for dim subvar 0 (and 2)
        # So stddevs[0, 1, 0] should be NaN (missing propagated)
        assert numpy.isnan(stddevs[0, 1, 0])
        # But stddevs[0, 1, 1] should be valid (different fact column)
        assert not numpy.isnan(stddevs[0, 1, 1])

    def test_2d_dim_2d_fact_with_missing_ignore_missing(self):
        """Stddev with 2D dim and 2D fact, ignore_missing=True."""
        fact_with_nan = self.fact_2d.copy()
        fact_with_nan[2, 0] = float("nan")

        stddevs = xcube([self.dim_2d]).stddev(fact_with_nan, ignore_missing=True)
        assert stddevs.shape == (3, 2, 2)

        # With ignore_missing=True, stddevs[0, 1, 0] should be computed
        # from rows [0, 4, 6] (excluding row 2 which has NaN)
        expected = numpy.std([1, 5, 7], ddof=1)
        assert arr_eq(stddevs[0, 1, 0], expected)

    def test_2d_dim_2d_fact_weighted_with_missing(self):
        """Weighted stddev with 2D dim and 2D fact, with missing values."""
        fact_with_nan = self.fact_2d.copy()
        fact_with_nan[2, 0] = float("nan")

        stddevs = xcube([self.dim_2d]).stddev(fact_with_nan, weights=self.weights)
        assert stddevs.shape == (3, 2, 2)

        # Missing should propagate
        assert numpy.isnan(stddevs[0, 1, 0])

    def test_2d_dim_2d_fact_weighted_ignore_missing(self):
        """Weighted stddev with 2D dim and 2D fact, ignore_missing=True."""
        fact_with_nan = self.fact_2d.copy()
        fact_with_nan[2, 0] = float("nan")

        stddevs = xcube([self.dim_2d]).stddev(
            fact_with_nan, weights=self.weights, ignore_missing=True
        )
        assert stddevs.shape == (3, 2, 2)

        # With ignore_missing=True, should have valid result
        assert not numpy.isnan(stddevs[0, 1, 0])
