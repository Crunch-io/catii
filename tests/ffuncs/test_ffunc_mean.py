import numpy

from catii import ccube, ffuncs, iindex

from .. import arr_eq

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


class TestFfuncMeanWorkflow:
    def _test_unweighted_workflow(self, arr):
        # Other tests often simply test output; here we test intermediate results.
        f = ffuncs.ffunc_mean(arr)
        arr, validity = ffuncs.as_separate_validity(arr)

        cube = ccube([])
        sums, counts, missings = f.get_initial_regions(cube)
        assert sums.tolist() == 12.0
        assert counts.tolist() == 4
        assert missings.tolist() == 1
        f.fill(cube, (sums, counts, missings))
        assert sums.tolist() == 12.0
        assert counts.tolist() == 4
        assert missings.tolist() == 1
        assert numpy.isnan(f.reduce(cube, (sums, counts, missings)))

        cube = ccube([idx1])
        sums, counts, missings = f.get_initial_regions(cube)
        assert sums.tolist() == [0.0, 0.0, 12.0]
        assert counts.tolist() == [0, 0, 4]
        assert missings.tolist() == [0, 0, 1]
        f.fill(cube, (sums, counts, missings))
        assert sums.tolist() == [0.0, 1.0, 12.0]
        assert counts.tolist() == [0, 1, 4]
        assert missings.tolist() == [0, 1, 1]
        assert arr_eq(
            f.reduce(cube, (sums, counts, missings)), [11 / 3.0, float("nan")]
        )

        cube = ccube([idx1, idx2])
        sums, counts, missings = f.get_initial_regions(cube)
        assert sums.tolist() == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 12.0]]
        assert counts.tolist() == [[0, 0, 0], [0, 0, 0], [0, 0, 4]]
        assert missings.tolist() == [[0, 0, 0], [0, 0, 0], [0, 0, 1]]
        f.fill(cube, (sums, counts, missings))
        assert sums.tolist() == [[0.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 5.0, 12.0]]
        assert counts.tolist() == [[0, 0, 0], [0, 1, 1], [0, 2, 4]]
        assert missings.tolist() == [[0, 0, 0], [0, 0, 1], [0, 0, 1]]
        assert arr_eq(
            f.reduce(cube, (sums, counts, missings)), [[3.5, 4.0], [float("nan"), 1.0]]
        )

    def _test_weighted_workflow(self, arr):
        # Other tests often simply test output; here we test intermediate results.
        f = ffuncs.ffunc_mean(arr, weights=wt)
        arr, arr_validity = ffuncs.as_separate_validity(arr)
        weight, wt_validity = ffuncs.as_separate_validity(wt)
        validity = arr_validity & wt_validity

        cube = ccube([idx1, idx2])
        sums, counts, missings = f.get_initial_regions(cube)
        assert sums.tolist() == [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, (arr[validity] * wt[validity]).sum()],
        ]
        assert counts.tolist() == [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, wt[validity].sum()],
        ]
        assert missings.tolist() == [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 2],
        ]
        f.fill(cube, (sums, counts, missings))
        assert sums.tolist() == [
            [0.0, 0.0, 0.0],
            [0.0, 0.25, (1.0 * 0.25)],
            [0.0, ((1.0 * 0.25) + (4.0 * 1.0)), (arr[validity] * wt[validity]).sum(),],
        ]
        assert counts.tolist() == [
            [0.0, 0.0, 0.0],  # idx1 == 0
            [0.0, 0.25, 0.25],  # idx1 == 1
            [0.0, (0.25 + 1.0), wt[validity].sum()],  # idx1 == any
        ]
        assert missings.tolist() == [
            [0, 0, 0],  # idx1 == 0
            [0, 0, 1],  # idx1 == 1
            [0, 0, 2],  # idx1 == any
        ]
        assert arr_eq(
            f.reduce(cube, (sums, counts, missings)),
            [[float("nan"), 4.0], [float("nan"), 1.0]],
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


class TestFfuncMeanIgnoreMissing:
    def test_ignore_missing(self):
        arr = [1.0, 2.0, float("nan"), 4.0, 5.0]

        # The cube of idx1 has rowids:
        # 0           1
        # [1, 3, 4]  [0, 2]
        # and arr values:
        # 0           1
        # [2.0, 4.0, 5.0]  [1.0, nan]

        means = ccube([idx1]).mean(arr)
        # Cell (1,) MUST be missing, because it had a missing input.
        assert arr_eq(means, [11 / 3.0, float("nan")])

        means = ccube([idx1]).mean(arr, ignore_missing=True)
        # Cell (1,) MUST NOT be missing, because it had a valid input.
        assert arr_eq(means, [11 / 3.0, 1.0])

        # The cube of idx1 x idx2 has arr values:
        #      ______idx 2______
        #      0           1
        # i|0  [2.0, 5.0]  [4.0]
        # d|
        # x|
        # 1|1  [nan]       [1.0]

        means = ccube([idx1, idx2]).mean(arr)
        # Cell (1, 0) MUST be missing, because it had a missing arr value.
        assert arr_eq(means, [[3.5, 4.0], [float("nan"), 1.0]])

        means = ccube([idx1, idx2]).mean(arr, ignore_missing=True)
        # Cell (1, 0) MUST be missing, because it had no non-missing inputs.
        assert arr_eq(means, [[3.5, 4.0], [float("nan"), 1.0]])


class TestFfuncMeanReturnValidity:
    def test_return_validity(self):
        arr = [1.0, 2.0, float("nan"), 4.0, 5.0]

        means = ccube([idx1]).mean(arr)
        assert arr_eq(means, [11 / 3.0, float("nan")])

        means, validity = ccube([idx1]).mean(arr, return_validity=True)
        assert means.tolist() == [11 / 3.0, 0.0]
        assert validity.tolist() == [True, False]

        means, validity = ccube([idx1]).mean(
            arr, ignore_missing=True, return_validity=True
        )
        assert means.tolist() == [11 / 3.0, 1.0]
        assert validity.tolist() == [True, True]

        means = ccube([idx1, idx2]).mean(arr)
        assert arr_eq(means, [[3.5, 4.0], [float("nan"), 1.0]])

        means, validity = ccube([idx1, idx2]).mean(arr, return_validity=True)
        assert means.tolist() == [[3.5, 4.0], [0.0, 1.0]]
        assert validity.tolist() == [[True, True], [False, True]]
