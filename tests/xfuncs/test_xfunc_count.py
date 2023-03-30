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


class TestXfuncCountWorkflow:
    def _test_unweighted_workflow(self):
        # Other tests often simply test output; here we test intermediate results.
        f = xfuncs.xfunc_count(N=5)

        cube = xcube([])
        (counts,) = f.get_initial_regions(cube)
        assert counts.tolist() == [0]
        coordinates = None
        f.fill(coordinates, (counts,))
        assert counts.tolist() == [5]
        assert f.reduce(cube, (counts,)) == [5]

        cube = xcube([arr1])
        (counts,) = f.get_initial_regions(cube)
        assert counts.tolist() == [0, 0]
        coordinates = reduce(operator.add, cube.strided_dims())
        f.fill(coordinates, (counts,))
        assert counts.tolist() == [3, 2]
        assert arr_eq(f.reduce(cube, (counts,)), [3, 2])

        cube = xcube([arr1, arr2])
        (counts,) = f.get_initial_regions(cube)
        assert counts.tolist() == [[0, 0], [0, 0]]
        coordinates = reduce(operator.add, cube.strided_dims())
        f.fill(coordinates, (counts,))
        assert counts.tolist() == [[2, 1], [1, 1]]
        assert arr_eq(f.reduce(cube, (counts,)), [[2, 1], [1, 1]])

    def _test_weighted_workflow(self):
        # Other tests often simply test output; here we test intermediate results.
        f = xfuncs.xfunc_count(weights=wt)
        weight, validity = xfuncs.as_separate_validity(wt)

        cube = xcube([arr1, arr2])
        counts, valids, missings = f.get_initial_regions(cube)
        assert counts.tolist() == [[0.0, 0.0], [0.0, 0.0]]
        assert valids.tolist() == [[0.0, 0.0], [0.0, 0.0]]
        assert missings.tolist() == [[0, 0], [0, 0]]
        coordinates = reduce(operator.add, cube.strided_dims())
        f.fill(coordinates, (counts, valids, missings))
        assert counts.tolist() == [[0.3, 1.0], [0.99, 0.25]]
        assert valids.tolist() == [[1, 1], [1, 1]]
        assert missings.tolist() == [[1, 0], [0, 0]]
        assert arr_eq(
            f.reduce(cube, (counts, valids, missings)),
            [[float("nan"), 1.0], [0.99, 0.25]],
        )

    def test_single_arr_with_nan_workflow(self):
        self._test_unweighted_workflow()
        self._test_weighted_workflow()


class TestXfuncCountWeights:
    def test_weights(self):
        counts = xcube([arr1]).count(weights=None)
        assert arr_eq(counts, [3, 2])

        counts = xcube([arr1]).count(weights=[0.25, 0.3, 0.99, 1.0, 0.5])
        assert arr_eq(counts, [1.8, 1.24])

        counts = xcube([arr1]).count(weights=[0.25, 0.3, 0.99, 1.0, float("nan")])
        assert arr_eq(counts, [float("nan"), 1.24])

        counts = xcube([arr1]).count(
            weights=[0.25, 0.3, 0.99, 1.0, float("nan")], ignore_missing=True
        )
        assert arr_eq(counts, [1.3, 1.24])

        counts = xcube([arr1]).count(weights=0.5)
        assert arr_eq(counts, [1.5, 1.0])

        counts = xcube([arr1]).count(weights=float("nan"))
        assert arr_eq(counts, [float("nan"), float("nan")])

        counts = xcube([arr1]).count(weights=float("nan"), ignore_missing=True)
        assert arr_eq(counts, [float("nan"), float("nan")])


class TestXfuncCountMissingness:
    # Make sure each combination of these options works properly:
    #  * ignore_missing = True/False
    #  * return_missing_as = NaN/(<sentinel value>, False)
    #  * at least one output cell which has no inputs
    #  * [count() takes no fact variable]
    #  * weights variable (with missings) as:
    #    * single array with NaN
    #    * no NaN, but separate "validity" array
    #    * includes NaN, but separate "validity" array
    arr_with_empty_cell = [0, 0, 0, 1, 0]

    dirty_weights = [9.0, 9.0, float("nan"), 9.0, 9.0]
    clean_weights = [9.0, 9.0, MAXFLOAT, 9.0, 9.0]
    weights_validity = [True, True, False, True, True]

    args = [
        dirty_weights,
        (clean_weights, weights_validity),
        (dirty_weights, weights_validity),
    ]

    @pytest.mark.parametrize("weights", args)
    def test_propagate_missing_return_nan(self, weights):
        WT = 9.0
        # The cube of arr1 has rowids:
        # 0           1
        # [1, 3, 4]  [0, 2]
        # and fact values:
        # 0           1
        # [2.0, 4.0, 5.0]  [1.0, nan]
        counts = xcube([arr1]).count(weights)
        # Cell (1,) MUST be missing, because it had a missing input.
        assert arr_eq(counts, [3.0 * WT, float("nan")])

        # The cube of arr1 x arr_with_empty_cell has fact values:
        #      _______idx w_______
        #      0           1
        # i|0  [2.0, 5.0]  [4.0]
        # d|
        # x|
        # 1|1  [1.0, nan]       []
        counts = xcube([arr1, self.arr_with_empty_cell]).count(weights)
        # Cell (1, 0) MUST be missing, because it had a missing input.
        # Cell (1, 1) MUST be missing, because it had no inputs.
        assert arr_eq(counts, [[2.0 * WT, 1.0 * WT], [float("nan"), float("nan")]])

    @pytest.mark.parametrize("weights", args)
    def test_ignore_missing_return_nan(self, weights):
        WT = 9.0
        # The cube of arr1 has rowids:
        # 0           1
        # [1, 3, 4]  [0, 2]
        # and fact values:
        # 0           1
        # [2.0, 4.0, 5.0]  [1.0, nan]
        counts = xcube([arr1]).count(weights, ignore_missing=True)
        # Cell (1,) MUST NOT be missing, because it had a valid input.
        assert arr_eq(counts, [3.0 * WT, 1.0 * WT])

        # The cube of arr1 x arr_with_empty_cell has fact values:
        #      _______idx w_______
        #      0           1
        # i|0  [2.0, 5.0]  [4.0]
        # d|
        # x|
        # 1|1  [1.0, nan]       []
        counts = xcube([arr1, self.arr_with_empty_cell]).count(
            weights, ignore_missing=True
        )
        # Cell (1, 0) MUST NOT be missing, because it had a valid input.
        # Cell (1, 1) MUST be missing, because it had no inputs.
        assert arr_eq(counts, [[2.0 * WT, 1.0 * WT], [1.0 * WT, float("nan")]])

    @pytest.mark.parametrize("weights", args)
    def test_propagate_missing_return_validity(self, weights):
        WT = 9.0
        # The cube of arr1 has rowids:
        # 0           1
        # [1, 3, 4]  [0, 2]
        # and fact values:
        # 0           1
        # [2.0, 4.0, 5.0]  [1.0, nan]
        counts, validity = xcube([arr1]).count(weights, return_missing_as=(-1, False))
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
        counts, validity = xcube([arr1, self.arr_with_empty_cell]).count(
            weights, return_missing_as=(-1, False)
        )
        # Cell (1, 0) MUST be missing, because it had a missing input.
        # Cell (1, 1) MUST be missing, because it had no inputs.
        assert arr_eq(counts, [[2.0 * WT, 1.0 * WT], [-1, -1]])
        assert arr_eq(validity, [[True, True], [False, False]])

    @pytest.mark.parametrize("weights", args)
    def test_ignore_missing_return_validity(self, weights):
        WT = 9.0
        # The cube of arr1 has rowids:
        # 0           1
        # [1, 3, 4]  [0, 2]
        # and fact values:
        # 0           1
        # [2.0, 4.0, 5.0]  [1.0, nan]
        counts, validity = xcube([arr1]).count(
            weights, ignore_missing=True, return_missing_as=(-1, False)
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
        counts, validity = xcube([arr1, self.arr_with_empty_cell]).count(
            weights, ignore_missing=True, return_missing_as=(-1, False)
        )
        # Cell (1, 0) MUST NOT be missing, because it had a valid input.
        # Cell (1, 1) MUST be missing, because it had no inputs.
        assert arr_eq(counts, [[2.0 * WT, 1.0 * WT], [1.0 * WT, -1]])
        assert arr_eq(validity, [[True, True], [True, False]])
