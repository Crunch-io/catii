import operator
import sys
from functools import reduce

import numpy
import pytest

from catii import xcube, xfuncs

from .. import arr_eq

MAXFLOAT = sys.float_info.max
arr1 = [1, 0, 1, 0, 0]  # iindex({(1,): [0, 2]}, 0, (5,))
arr2 = [1, 0, 0, 1, 0]  # iindex({(1,): [0, 3]}, 0, (5,))
wt = numpy.array([0.25, 0.3, 0.99, 1.0, float("nan")])

# The cube of arr1 x arr2 has rowids:
#      ___arr 2___
#      0       1
# a|0  [1, 4]  [3]
# r|
# r|
# 1|1  [2]     [0]
#
# and therefore weights:
#      ______arr 2_______
#      0           1
# a|0  [0.3, nan]  [1.0]
# r|
# r|
# 1|1  [0.99       [0.25]


class TestXfuncValidCountWorkflow:
    def _test_unweighted_workflow(self, factvar):
        # Other tests often simply test output; here we test intermediate results.
        f = xfuncs.xfunc_valid_count(factvar)

        cube = xcube([])
        counts, valids, missings = f.get_initial_regions(cube)
        assert counts.tolist() == [0]
        assert valids.tolist() == [0]
        assert missings.tolist() == [0]
        coordinates = None
        f.fill(coordinates, (counts, valids, missings))
        assert counts.tolist() == [4]
        assert valids.tolist() == [4]
        assert missings.tolist() == [1]
        assert numpy.isnan(f.reduce(cube, (counts, valids, missings)))

        cube = xcube([arr1])
        counts, valids, missings = f.get_initial_regions(cube)
        assert counts.tolist() == [0, 0]
        assert valids.tolist() == [0, 0]
        assert missings.tolist() == [0, 0]
        coordinates = reduce(operator.add, cube.strided_dims())
        f.fill(coordinates, (counts, valids, missings))
        assert counts.tolist() == [3, 1]
        assert valids.tolist() == [3, 1]
        assert missings.tolist() == [0, 1]
        assert arr_eq(f.reduce(cube, (counts, valids, missings)), [3, float("nan")])

        cube = xcube([arr1, arr2])
        counts, valids, missings = f.get_initial_regions(cube)
        assert counts.tolist() == [[0, 0], [0, 0]]
        assert valids.tolist() == [[0, 0], [0, 0]]
        assert missings.tolist() == [[0, 0], [0, 0]]
        coordinates = reduce(operator.add, cube.strided_dims())
        f.fill(coordinates, (counts, valids, missings))
        assert counts.tolist() == [[2, 1], [0, 1]]
        assert valids.tolist() == [[2, 1], [0, 1]]
        assert missings.tolist() == [[0, 0], [1, 0]]
        assert arr_eq(
            f.reduce(cube, (counts, valids, missings)), [[2, 1], [float("nan"), 1]]
        )

    def _test_weighted_workflow(self, factvar):
        # Other tests often simply test output; here we test intermediate results.
        f = xfuncs.xfunc_valid_count(factvar, weights=wt)

        cube = xcube([arr1, arr2])
        counts, valids, missings = f.get_initial_regions(cube)
        assert counts.tolist() == [[0.0, 0.0], [0.0, 0.0]]
        assert valids.tolist() == [[0, 0], [0, 0]]
        assert missings.tolist() == [[0, 0], [0, 0]]
        coordinates = reduce(operator.add, cube.strided_dims())
        f.fill(coordinates, (counts, valids, missings))
        assert counts.tolist() == [[0.3, 1.0], [0.0, 0.25]]
        assert valids.tolist() == [[1, 1], [0, 1]]
        assert missings.tolist() == [[1, 0], [1, 0]]
        assert arr_eq(
            f.reduce(cube, (counts, valids, missings)),
            [[float("nan"), 1.0], [float("nan"), 0.25]],
        )

    def test_single_arr_with_nan_workflow(self):
        factvar = [1.0, 2.0, float("nan"), 4.0, 5.0]
        self._test_unweighted_workflow(factvar)
        self._test_weighted_workflow(factvar)

    def test_tuple_arr_workflow(self):
        factvar = ([1.0, 2.0, 3.0, 4.0, 5.0], [True, True, False, True, True])
        self._test_unweighted_workflow(factvar)
        self._test_weighted_workflow(factvar)

    def test_tuple_arr_with_nan_workflow(self):
        factvar = ([1.0, 2.0, float("nan"), 4.0, 5.0], [True, True, False, True, True])
        self._test_unweighted_workflow(factvar)
        self._test_weighted_workflow(factvar)


class TestXfuncValidCountMissingness:
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
    arr_with_empty_cell = [0, 0, 0, 1, 0]

    dirty_fact = [1.0, 2.0, float("nan"), 4.0, 5.0]
    clean_fact = [1.0, 2.0, MAXFLOAT, 4.0, 5.0]
    fact_validity = [True, True, False, True, True]
    fact_all_valid = [True] * 5

    dirty_weights = [9.0, 9.0, float("nan"), 9.0, 9.0]
    clean_weights = [9.0, 9.0, MAXFLOAT, 9.0, 9.0]
    weights_validity = [True, True, False, True, True]

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
        WT = 1.0 if weights is None else 9.0
        # The cube of arr1 has rowids:
        # 0           1
        # [1, 3, 4]  [0, 2]
        # and fact values:
        # 0           1
        # [2.0, 4.0, 5.0]  [1.0, nan]
        counts = xcube([arr1]).valid_count(factvar, weights)
        # Cell (1,) MUST be missing, because it had a missing input.
        assert arr_eq(counts, [3.0 * WT, float("nan")])

        # The cube of arr1 x arr_with_empty_cell has fact values:
        #      _______idx w_______
        #      0           1
        # i|0  [2.0, 5.0]  [4.0]
        # d|
        # x|
        # 1|1  [1.0, nan]       []
        counts = xcube([arr1, self.arr_with_empty_cell]).valid_count(factvar, weights)
        # Cell (1, 0) MUST be missing, because it had a missing input.
        # Cell (1, 1) MUST be missing, because it had no inputs.
        assert arr_eq(counts, [[2.0 * WT, 1.0 * WT], [float("nan"), float("nan")]])

    @pytest.mark.parametrize("factvar,weights", params)
    def test_ignore_missing_return_nan(self, factvar, weights):
        WT = 1.0 if weights is None else 9.0
        # The cube of arr1 has rowids:
        # 0           1
        # [1, 3, 4]  [0, 2]
        # and fact values:
        # 0           1
        # [2.0, 4.0, 5.0]  [1.0, nan]
        counts = xcube([arr1]).valid_count(factvar, weights, ignore_missing=True)
        # Cell (1,) MUST NOT be missing, because it had a valid input.
        assert arr_eq(counts, [3.0 * WT, 1.0 * WT])

        # The cube of arr1 x arr_with_empty_cell has fact values:
        #      _______idx w_______
        #      0           1
        # i|0  [2.0, 5.0]  [4.0]
        # d|
        # x|
        # 1|1  [1.0, nan]       []
        counts = xcube([arr1, self.arr_with_empty_cell]).valid_count(
            factvar, weights, ignore_missing=True
        )
        # Cell (1, 0) MUST NOT be missing, because it had a valid input.
        # Cell (1, 1) MUST be missing, because it had no inputs.
        assert arr_eq(counts, [[2.0 * WT, 1.0 * WT], [1.0 * WT, float("nan")]])

    @pytest.mark.parametrize("factvar,weights", params)
    def test_propagate_missing_return_validity(self, factvar, weights):
        WT = 1.0 if weights is None else 9.0
        # The cube of arr1 has rowids:
        # 0           1
        # [1, 3, 4]  [0, 2]
        # and fact values:
        # 0           1
        # [2.0, 4.0, 5.0]  [1.0, nan]
        counts, validity = xcube([arr1]).valid_count(
            factvar, weights, return_missing_as=(-1, False)
        )
        # Cell (1,) MUST be missing, because it had a missing input.
        assert arr_eq(counts, [3.0 * WT, -1])
        assert arr_eq(validity, [True, False])

        # The cube of arr1 x arr_with_empty_cell has fact values:
        #      _______idx w_______
        #      0           1
        # i|0  [2.0, 5.0]  [4.0]
        # d|
        # x|
        # 1|1  [1.0, nan]       []
        counts, validity = xcube([arr1, self.arr_with_empty_cell]).valid_count(
            factvar, weights, return_missing_as=(-1, False)
        )
        # Cell (1, 0) MUST be missing, because it had a missing input.
        # Cell (1, 1) MUST be missing, because it had no inputs.
        assert arr_eq(counts, [[2.0 * WT, 1.0 * WT], [-1, -1]])
        assert arr_eq(validity, [[True, True], [False, False]])

    @pytest.mark.parametrize("factvar,weights", params)
    def test_ignore_missing_return_validity(self, factvar, weights):
        WT = 1.0 if weights is None else 9.0
        # The cube of arr1 has rowids:
        # 0           1
        # [1, 3, 4]  [0, 2]
        # and fact values:
        # 0           1
        # [2.0, 4.0, 5.0]  [1.0, nan]
        counts, validity = xcube([arr1]).valid_count(
            factvar, weights, ignore_missing=True, return_missing_as=(-1, False)
        )
        # Cell (1,) MUST NOT be missing, because it had a valid input.
        assert arr_eq(counts, [3.0 * WT, 1.0 * WT])
        assert arr_eq(validity, [True, True])

        # The cube of arr1 x arr_with_empty_cell has fact values:
        #      _______idx w_______
        #      0           1
        # i|0  [2.0, 5.0]  [4.0]
        # d|
        # x|
        # 1|1  [1.0, nan]       []
        counts, validity = xcube([arr1, self.arr_with_empty_cell]).valid_count(
            factvar, weights, ignore_missing=True, return_missing_as=(-1, False)
        )
        # Cell (1, 0) MUST NOT be missing, because it had a valid input.
        # Cell (1, 1) MUST be missing, because it had no inputs.
        assert arr_eq(counts, [[2.0 * WT, 1.0 * WT], [1.0 * WT, -1]])
        assert arr_eq(validity, [[True, True], [True, False]])


class TestXfuncValidCountFactVar:
    def test_2d_factvar(self):
        factvar = [
            [1.0, 10.0],
            [2.0, 11.0],
            [float("nan"), 12.0],
            [4.0, 13.0],
            [5.0, 14.0],
        ]

        counts = xcube([arr1]).valid_count(factvar)
        assert arr_eq(counts, [[3, 3], [float("nan"), 2]])

        counts = xcube([arr1, arr2]).valid_count(factvar)
        assert arr_eq(counts, [[[2, 2], [1, 1]], [[float("nan"), 1], [1, 1]]])
