import sys

import numpy
import pytest

from catii import ccube, ffuncs, iindex

from .. import arr_eq, compare_ccube_to_xcube

MAXFLOAT = sys.float_info.max
idx1 = iindex({(1,): [0, 2]}, 0, (5,))  # [1, 0, 1, 0, 0]
idx2 = iindex({(1,): [0, 3]}, 0, (5,))  # [1, 0, 0, 1, 0]
wt = numpy.array([0.25, 0.3, 0.99, 1.0, float("nan")])

# The cube of idx1 x idx2 has rowids:
#      ___idx 2___
#      0       1
# i|0  [1, 4]  [3]
# d|
# x|
# 1|1  [2]     [0]
#
# and therefore weights:
#      ______idx 2_______
#      0           1
# i|0  [0.3, nan]  [1.0]
# d|
# x|
# 1|1  [0.99       [0.25]


class TestFfuncCountWorkflow:
    def _test_unweighted_workflow(self):
        # Other tests often simply test output; here we test intermediate results.
        f = ffuncs.ffunc_count(N=5)

        cube = ccube([])
        (counts,) = f.get_initial_regions(cube)
        assert counts.tolist() == 5
        cube.walk(f.fill_func((counts,)))
        assert counts.tolist() == 5
        assert f.reduce(cube, (counts,)) == 5

        cube = ccube([idx1])
        (counts,) = f.get_initial_regions(cube)
        assert counts.tolist() == [0, 0, 5]
        cube.walk(f.fill_func((counts,)))
        assert counts.tolist() == [0, 2, 5]
        assert arr_eq(f.reduce(cube, (counts,)), [3, 2])

        cube = ccube([idx1, idx2])
        (counts,) = f.get_initial_regions(cube)
        assert counts.tolist() == [[0, 0, 0], [0, 0, 0], [0, 0, 5]]
        cube.walk(f.fill_func((counts,)))
        assert counts.tolist() == [[0, 0, 0], [0, 1, 2], [0, 2, 5]]
        assert arr_eq(f.reduce(cube, (counts,)), [[2, 1], [1, 1]])

    def _test_weighted_workflow(self):
        # Other tests often simply test output; here we test intermediate results.
        f = ffuncs.ffunc_count(weights=wt)
        weight, validity = ffuncs.as_separate_validity(wt)

        cube = ccube([idx1, idx2])
        counts, valids, missings = f.get_initial_regions(cube)
        assert counts.tolist() == [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, wt[validity].sum()],
        ]
        assert valids.tolist() == [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, validity.sum()],
        ]
        assert missings.tolist() == [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 1],
        ]
        cube.walk(f.fill_func((counts, valids, missings)))
        assert counts.tolist() == [
            [0.0, 0.0, 0.0],
            [0.0, 0.25, (0.25 + 0.99)],
            [0.0, (0.25 + 1.0), wt[validity].sum()],
        ]
        assert valids.tolist() == [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 2.0],
            [0.0, 2.0, validity.sum()],
        ]
        assert missings.tolist() == [
            [0, 0, 0],  # idx1 == 0
            [0, 0, 0],  # idx1 == 1
            [0, 0, 1],  # idx1 == any
        ]
        assert arr_eq(
            f.reduce(cube, (counts, valids, missings)),
            [[float("nan"), 1.0], [0.99, 0.25]],
        )

    def test_single_arr_with_nan_workflow(self):
        self._test_unweighted_workflow()
        self._test_weighted_workflow()


class TestFfuncCountWeights:
    def test_weights(self):
        with compare_ccube_to_xcube():
            counts = ccube([idx1]).count(weights=None)
            assert arr_eq(counts, [3, 2])

            counts = ccube([idx1]).count(weights=[0.25, 0.3, 0.99, 1.0, 0.5])
            assert arr_eq(counts, [1.8, 1.24])

            counts = ccube([idx1]).count(weights=[0.25, 0.3, 0.99, 1.0, float("nan")])
            assert arr_eq(counts, [float("nan"), 1.24])

            counts = ccube([idx1]).count(
                weights=[0.25, 0.3, 0.99, 1.0, float("nan")], ignore_missing=True
            )
            assert arr_eq(counts, [1.3, 1.24])

            counts = ccube([idx1]).count(weights=0.5)
            assert arr_eq(counts, [1.5, 1.0])

            counts = ccube([idx1]).count(weights=float("nan"))
            assert arr_eq(counts, [float("nan"), float("nan")])

            counts = ccube([idx1]).count(weights=float("nan"), ignore_missing=True)
            assert arr_eq(counts, [float("nan"), float("nan")])


class TestFfuncCountMissingness:
    # Make sure each combination of these options works properly:
    #  * ignore_missing = True/False
    #  * return_missing_as = NaN/(<sentinel value>, False)
    #  * at least one output cell which has no inputs
    #  * [count() takes no fact variable]
    #  * weights variable (with missings) as:
    #    * single array with NaN
    #    * no NaN, but separate "validity" array
    #    * includes NaN, but separate "validity" array
    idx_with_empty_cell = iindex({(1,): [3]}, 0, (5,))  # [0, 0, 0, 1, 0]

    dirty_weights = [9.0, 9.0, float("nan"), 9.0, 9.0]
    clean_weights = [9.0, 9.0, MAXFLOAT, 9.0, 9.0]
    weights_validity = [True, True, False, True, True]

    params = [
        dirty_weights,
        (clean_weights, weights_validity),
        (dirty_weights, weights_validity),
    ]

    @pytest.mark.parametrize("weights", params)
    def test_propagate_missing_return_nan(self, weights):
        WT = 9.0
        with compare_ccube_to_xcube():
            # The cube of idx1 has rowids:
            # 0           1
            # [1, 3, 4]  [0, 2]
            # and fact values:
            # 0           1
            # [2.0, 4.0, 5.0]  [1.0, nan]
            counts = ccube([idx1]).count(weights)
            # Cell (1,) MUST be missing, because it had a missing input.
            assert arr_eq(counts, [3.0 * WT, float("nan")])

            # The cube of idx1 x idx_with_empty_cell has fact values:
            #      _______idx w_______
            #      0           1
            # i|0  [2.0, 5.0]  [4.0]
            # d|
            # x|
            # 1|1  [1.0, nan]       []
            counts = ccube([idx1, self.idx_with_empty_cell]).count(weights)
            # Cell (1, 0) MUST be missing, because it had a missing input.
            # Cell (1, 1) MUST be missing, because it had no inputs.
            assert arr_eq(counts, [[2.0 * WT, 1.0 * WT], [float("nan"), float("nan")]])

    @pytest.mark.parametrize("weights", params)
    def test_ignore_missing_return_nan(self, weights):
        WT = 9.0
        with compare_ccube_to_xcube():
            # The cube of idx1 has rowids:
            # 0           1
            # [1, 3, 4]  [0, 2]
            # and fact values:
            # 0           1
            # [2.0, 4.0, 5.0]  [1.0, nan]
            counts = ccube([idx1]).count(weights, ignore_missing=True)
            # Cell (1,) MUST NOT be missing, because it had a valid input.
            assert arr_eq(counts, [3.0 * WT, 1.0 * WT])

            # The cube of idx1 x idx_with_empty_cell has fact values:
            #      _______idx w_______
            #      0           1
            # i|0  [2.0, 5.0]  [4.0]
            # d|
            # x|
            # 1|1  [1.0, nan]       []
            counts = ccube([idx1, self.idx_with_empty_cell]).count(
                weights, ignore_missing=True
            )
            # Cell (1, 0) MUST NOT be missing, because it had a valid input.
            # Cell (1, 1) MUST be missing, because it had no inputs.
            assert arr_eq(counts, [[2.0 * WT, 1.0 * WT], [1.0 * WT, float("nan")]])

    @pytest.mark.parametrize("weights", params)
    def test_propagate_missing_return_validity(self, weights):
        WT = 9.0
        with compare_ccube_to_xcube():
            # The cube of idx1 has rowids:
            # 0           1
            # [1, 3, 4]  [0, 2]
            # and fact values:
            # 0           1
            # [2.0, 4.0, 5.0]  [1.0, nan]
            counts, validity = ccube([idx1]).count(
                weights, return_missing_as=(-1, False)
            )
            # Cell (1,) MUST be missing, because it had a missing input.
            assert arr_eq(counts, [3.0 * WT, -1])
            assert arr_eq(validity, [True, False])

            # The cube of idx1 x idx_with_empty_cell has fact values:
            #      _______idx w_______
            #      0           1
            # i|0  [2.0, 5.0]  [4.0]
            # d|
            # x|
            # 1|1  [1.0, nan]       []
            counts, validity = ccube([idx1, self.idx_with_empty_cell]).count(
                weights, return_missing_as=(-1, False)
            )
            # Cell (1, 0) MUST be missing, because it had a missing input.
            # Cell (1, 1) MUST be missing, because it had no inputs.
            assert arr_eq(counts, [[2.0 * WT, 1.0 * WT], [-1, -1]])
            assert arr_eq(validity, [[True, True], [False, False]])

    @pytest.mark.parametrize("weights", params)
    def test_ignore_missing_return_validity(self, weights):
        WT = 9.0
        with compare_ccube_to_xcube():
            # The cube of idx1 has rowids:
            # 0           1
            # [1, 3, 4]  [0, 2]
            # and fact values:
            # 0           1
            # [2.0, 4.0, 5.0]  [1.0, nan]
            counts, validity = ccube([idx1]).count(
                weights, ignore_missing=True, return_missing_as=(-1, False)
            )
            # Cell (1,) MUST NOT be missing, because it had a valid input.
            assert arr_eq(counts, [3.0 * WT, 1.0 * WT])
            assert arr_eq(validity, [True, True])

            # The cube of idx1 x idx_with_empty_cell has fact values:
            #      _______idx w_______
            #      0           1
            # i|0  [2.0, 5.0]  [4.0]
            # d|
            # x|
            # 1|1  [1.0, nan]       []
            counts, validity = ccube([idx1, self.idx_with_empty_cell]).count(
                weights, ignore_missing=True, return_missing_as=(-1, False)
            )
            # Cell (1, 0) MUST NOT be missing, because it had a valid input.
            # Cell (1, 1) MUST be missing, because it had no inputs.
            assert arr_eq(counts, [[2.0 * WT, 1.0 * WT], [1.0 * WT, -1]])
            assert arr_eq(validity, [[True, True], [True, False]])
