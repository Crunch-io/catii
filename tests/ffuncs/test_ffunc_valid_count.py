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


class TestFfuncValidCountWorkflow:
    def _test_unweighted_workflow(self, arr):
        # Other tests often simply test output; here we test intermediate results.
        f = ffuncs.ffunc_valid_count(arr)
        arr, validity = ffuncs.as_separate_validity(arr)

        cube = ccube([])
        counts, valids, missings = f.get_initial_regions(cube)
        assert counts.tolist() == 4
        assert valids.tolist() == 4
        assert missings.tolist() == 1
        cube.walk(f.fill_func((counts, valids, missings)))
        assert counts.tolist() == 4
        assert valids.tolist() == 4
        assert missings.tolist() == 1
        assert numpy.isnan(f.reduce(cube, (counts, valids, missings)))

        cube = ccube([idx1])
        counts, valids, missings = f.get_initial_regions(cube)
        assert counts.tolist() == [0, 0, 4]
        assert valids.tolist() == [0, 0, 4]
        assert missings.tolist() == [0, 0, 1]
        cube.walk(f.fill_func((counts, valids, missings)))
        assert counts.tolist() == [0, 1, 4]
        assert valids.tolist() == [0, 1, 4]
        assert missings.tolist() == [0, 1, 1]
        assert arr_eq(f.reduce(cube, (counts, valids, missings)), [3, float("nan")])

        cube = ccube([idx1, idx2])
        counts, valids, missings = f.get_initial_regions(cube)
        assert counts.tolist() == [[0, 0, 0], [0, 0, 0], [0, 0, 4]]
        assert valids.tolist() == [[0, 0, 0], [0, 0, 0], [0, 0, 4]]
        assert missings.tolist() == [[0, 0, 0], [0, 0, 0], [0, 0, 1]]
        cube.walk(f.fill_func((counts, valids, missings)))
        assert counts.tolist() == [[0, 0, 0], [0, 1, 1], [0, 2, 4]]
        assert valids.tolist() == [[0, 0, 0], [0, 1, 1], [0, 2, 4]]
        assert missings.tolist() == [[0, 0, 0], [0, 0, 1], [0, 0, 1]]
        assert arr_eq(
            f.reduce(cube, (counts, valids, missings)), [[2, 1], [float("nan"), 1]]
        )

    def _test_weighted_workflow(self, arr):
        # Other tests often simply test output; here we test intermediate results.
        f = ffuncs.ffunc_valid_count(arr, weights=wt)
        arr, arr_validity = ffuncs.as_separate_validity(arr)
        weight, wt_validity = ffuncs.as_separate_validity(wt)
        validity = arr_validity & wt_validity

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
            [0, 0, 2],
        ]
        cube.walk(f.fill_func((counts, valids, missings)))
        assert counts.tolist() == [
            [0.0, 0.0, 0.0],
            [0.0, 0.25, 0.25],
            [0.0, (0.25 + 1.0), wt[validity].sum()],
        ]
        assert valids.tolist() == [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0],
            [0.0, 2.0, validity.sum()],
        ]
        assert missings.tolist() == [
            [0, 0, 0],  # idx1 == 0
            [0, 0, 1],  # idx1 == 1
            [0, 0, 2],  # idx1 == any
        ]
        assert arr_eq(
            f.reduce(cube, (counts, valids, missings)),
            [[float("nan"), 1.0], [float("nan"), 0.25]],
        )

    def test_single_arr_with_nan_workflow(self):
        arr = [1.0, 2.0, float("nan"), 4.0, 5.0]
        self._test_unweighted_workflow(arr)
        self._test_weighted_workflow(arr)

    def test_tuple_arr_workflow(self):
        arr = ([1.0, 2.0, 3.0, 4.0, 5.0], [True, True, False, True, True])
        self._test_unweighted_workflow(arr)
        self._test_weighted_workflow(arr)

    def test_tuple_arr_with_nan_workflow(self):
        arr = ([1.0, 2.0, float("nan"), 4.0, 5.0], [True, True, False, True, True])
        self._test_unweighted_workflow(arr)
        self._test_weighted_workflow(arr)


class TestFfuncValidCountMissingness:
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
    idx_with_empty_cell = iindex({(1,): [3]}, 0, (5,))  # [0, 0, 0, 1, 0]

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
        with compare_ccube_to_xcube():
            # The cube of idx1 has rowids:
            # 0           1
            # [1, 3, 4]  [0, 2]
            # and fact values:
            # 0           1
            # [2.0, 4.0, 5.0]  [1.0, nan]
            counts = ccube([idx1]).valid_count(factvar, weights)
            # Cell (1,) MUST be missing, because it had a missing input.
            assert arr_eq(counts, [3.0 * WT, float("nan")])

            # The cube of idx1 x idx_with_empty_cell has fact values:
            #      _______idx w_______
            #      0           1
            # i|0  [2.0, 5.0]  [4.0]
            # d|
            # x|
            # 1|1  [1.0, nan]       []
            counts = ccube([idx1, self.idx_with_empty_cell]).valid_count(
                factvar, weights
            )
            # Cell (1, 0) MUST be missing, because it had a missing input.
            # Cell (1, 1) MUST be missing, because it had no inputs.
            assert arr_eq(counts, [[2.0 * WT, 1.0 * WT], [float("nan"), float("nan")]])

    @pytest.mark.parametrize("factvar,weights", params)
    def test_ignore_missing_return_nan(self, factvar, weights):
        WT = 1.0 if weights is None else 9.0
        with compare_ccube_to_xcube():
            # The cube of idx1 has rowids:
            # 0           1
            # [1, 3, 4]  [0, 2]
            # and fact values:
            # 0           1
            # [2.0, 4.0, 5.0]  [1.0, nan]
            counts = ccube([idx1]).valid_count(factvar, weights, ignore_missing=True)
            # Cell (1,) MUST NOT be missing, because it had a valid input.
            assert arr_eq(counts, [3.0 * WT, 1.0 * WT])

            # The cube of idx1 x idx_with_empty_cell has fact values:
            #      _______idx w_______
            #      0           1
            # i|0  [2.0, 5.0]  [4.0]
            # d|
            # x|
            # 1|1  [1.0, nan]       []
            counts = ccube([idx1, self.idx_with_empty_cell]).valid_count(
                factvar, weights, ignore_missing=True
            )
            # Cell (1, 0) MUST NOT be missing, because it had a valid input.
            # Cell (1, 1) MUST be missing, because it had no inputs.
            assert arr_eq(counts, [[2.0 * WT, 1.0 * WT], [1.0 * WT, float("nan")]])

    @pytest.mark.parametrize("factvar,weights", params)
    def test_propagate_missing_return_validity(self, factvar, weights):
        WT = 1.0 if weights is None else 9.0
        with compare_ccube_to_xcube():
            # The cube of idx1 has rowids:
            # 0           1
            # [1, 3, 4]  [0, 2]
            # and fact values:
            # 0           1
            # [2.0, 4.0, 5.0]  [1.0, nan]
            counts, validity = ccube([idx1]).valid_count(
                factvar, weights, return_missing_as=(-1, False)
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
            counts, validity = ccube([idx1, self.idx_with_empty_cell]).valid_count(
                factvar, weights, return_missing_as=(-1, False)
            )
            # Cell (1, 0) MUST be missing, because it had a missing input.
            # Cell (1, 1) MUST be missing, because it had no inputs.
            assert arr_eq(counts, [[2.0 * WT, 1.0 * WT], [-1, -1]])
            assert arr_eq(validity, [[True, True], [False, False]])

    @pytest.mark.parametrize("factvar,weights", params)
    def test_ignore_missing_return_validity(self, factvar, weights):
        WT = 1.0 if weights is None else 9.0
        with compare_ccube_to_xcube():
            # The cube of idx1 has rowids:
            # 0           1
            # [1, 3, 4]  [0, 2]
            # and fact values:
            # 0           1
            # [2.0, 4.0, 5.0]  [1.0, nan]
            counts, validity = ccube([idx1]).valid_count(
                factvar, weights, ignore_missing=True, return_missing_as=(-1, False)
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
            counts, validity = ccube([idx1, self.idx_with_empty_cell]).valid_count(
                factvar, weights, ignore_missing=True, return_missing_as=(-1, False)
            )
            # Cell (1, 0) MUST NOT be missing, because it had a valid input.
            # Cell (1, 1) MUST be missing, because it had no inputs.
            assert arr_eq(counts, [[2.0 * WT, 1.0 * WT], [1.0 * WT, -1]])
            assert arr_eq(validity, [[True, True], [True, False]])
