import numpy

from catii import ccube, ffuncs, iindex

from .. import arr_eq

idx1 = iindex({(1,): [0, 2]}, 0, (5,))  # [1, 0, 1, 0, 0]
idx2 = iindex({(1,): [0, 3]}, 0, (5,))  # [1, 0, 0, 1, 0]
wt = numpy.array([0.25, 0.3, 0.99, 1.0, float("nan")])

# The cube of idx1 x idx2 has rowids:
#      ___idx 1___
#      0       1
# i|0  [1, 5]  [2]
# d|
# x|
# 2|1  [3]     [0]
#
# and therefore weights:
#      ______idx 1_______
#      0           1
# i|0  [0.3, nan]  [0.99]
# d|
# x|
# 2|1  [1.0]       [0.25]


class TestFfuncSumWorkflow:
    def _test_unweighted_workflow(self, arr):
        # Other tests often simply test output; here we test intermediate results.
        f = ffuncs.ffunc_sum(arr)
        arr, validity = ffuncs.as_separate_validity(arr)

        cube = ccube([])
        sums, missings = f.get_initial_regions(cube)
        assert sums.tolist() == 12.0
        assert missings.tolist() == 1
        f.fill(cube, (sums, missings))
        assert sums.tolist() == 12.0
        assert missings.tolist() == 1
        assert numpy.isnan(f.reduce(cube, (sums, missings)))

        cube = ccube([idx1])
        sums, missings = f.get_initial_regions(cube)
        assert sums.tolist() == [0.0, 0.0, 12.0]
        assert missings.tolist() == [0, 0, 1]
        f.fill(cube, (sums, missings))
        assert sums.tolist() == [0.0, 1.0, 12.0]
        assert missings.tolist() == [0, 1, 1]
        assert arr_eq(f.reduce(cube, (sums, missings)), [11, float("nan")])

        cube = ccube([idx1, idx2])
        sums, missings = f.get_initial_regions(cube)
        assert sums.tolist() == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 12.0]]
        assert missings.tolist() == [[0, 0, 0], [0, 0, 0], [0, 0, 1]]
        f.fill(cube, (sums, missings))
        assert sums.tolist() == [[0.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 5.0, 12.0]]
        assert missings.tolist() == [[0, 0, 0], [0, 0, 1], [0, 0, 1]]
        assert arr_eq(
            f.reduce(cube, (sums, missings)), [[7.0, 4.0], [float("nan"), 1.0]]
        )

    def _test_weighted_workflow(self, arr):
        # Other tests often simply test output; here we test intermediate results.
        f = ffuncs.ffunc_sum(arr, weights=wt)
        arr, arr_validity = ffuncs.as_separate_validity(arr)
        weight, wt_validity = ffuncs.as_separate_validity(wt)
        validity = arr_validity & wt_validity

        cube = ccube([idx1, idx2])
        sums, missings = f.get_initial_regions(cube)
        assert sums.tolist() == [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, (arr[validity] * wt[validity]).sum()],
        ]
        assert missings.tolist() == [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 2],
        ]
        f.fill(cube, (sums, missings))
        assert sums.tolist() == [
            [0.0, 0.0, 0.0],
            [0.0, 0.25, (1.0 * 0.25)],
            [0.0, ((1.0 * 0.25) + (4.0 * 1.0)), (arr[validity] * wt[validity]).sum()],
        ]
        assert missings.tolist() == [
            [0, 0, 0],  # idx1 == 0
            [0, 0, 1],  # idx1 == 1
            [0, 0, 2],  # idx1 == any
        ]
        assert arr_eq(
            f.reduce(cube, (sums, missings)),
            [[float("nan"), 4.0], [float("nan"), 0.25]],
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


class TestFfuncSumWeights:
    def test_weights(self):
        arr = [1.0, 2.0, 3.0, 4.0, 5.0]

        sums = ccube([idx1]).sum(arr, weights=None)
        assert arr_eq(sums, [11.0, 4.0])

        sums = ccube([idx1]).sum(arr, weights=[0.25, 0.3, 0.99, 1.0, 0.5])
        assert arr_eq(sums, [7.1, 3.22])

        sums = ccube([idx1]).sum(arr, weights=[0.25, 0.3, 0.99, 1.0, float("nan")])
        assert arr_eq(sums, [float("nan"), 3.22])

        sums = ccube([idx1]).sum(
            arr, weights=[0.25, 0.3, 0.99, 1.0, float("nan")], ignore_missing=True
        )
        assert arr_eq(sums, [4.6, 3.22])

        sums = ccube([idx1]).sum(arr, weights=0.5)
        assert arr_eq(sums, [5.5, 2.0])

        sums = ccube([idx1]).sum(arr, weights=float("nan"))
        assert arr_eq(sums, [float("nan"), float("nan")])

        sums = ccube([idx1]).sum(arr, weights=float("nan"), ignore_missing=True)
        assert arr_eq(sums, [float("nan"), float("nan")])


class TestFfuncSumIgnoreMissing:
    def test_ignore_missing(self):
        arr = [1.0, 2.0, float("nan"), 4.0, 5.0]

        # The cube of idx1 has rowids:
        # 0           1
        # [1, 3, 4]  [0, 2]
        # and arr values:
        # 0           1
        # [2.0, 4.0, 5.0]  [1.0, nan]

        sums = ccube([idx1]).sum(arr)
        # Cell (1,) MUST be missing, because it had a missing input.
        assert arr_eq(sums, [11.0, float("nan")])

        sums = ccube([idx1]).sum(arr, ignore_missing=True)
        # Cell (1,) MUST NOT be missing, because it had a valid input.
        assert arr_eq(sums, [11.0, 1.0])

        # The cube of idx1 x idx2 has arr values:
        #      ______idx 2______
        #      0           1
        # i|0  [2.0, 5.0]  [4.0]
        # d|
        # x|
        # 1|1  [nan]       [1.0]

        sums = ccube([idx1, idx2]).sum(arr)
        # Cell (1, 0) MUST be missing, because it had a missing arr value.
        assert arr_eq(sums, [[7.0, 4.0], [float("nan"), 1.0]])

        sums = ccube([idx1, idx2]).sum(arr, ignore_missing=True)
        # Cell (1, 0) MUST be missing, because it had no non-missing inputs.
        assert arr_eq(sums, [[7.0, 4.0], [float("nan"), 1.0]])


class TestFfuncSumReturnValidity:
    def test_return_validity(self):
        arr = [1.0, 2.0, float("nan"), 4.0, 5.0]

        sums = ccube([idx1]).sum(arr)
        assert arr_eq(sums, [11, float("nan")])

        sums, validity = ccube([idx1]).sum(arr, return_validity=True)
        assert sums.tolist() == [11, 0.0]
        assert validity.tolist() == [True, False]

        sums, validity = ccube([idx1]).sum(
            arr, ignore_missing=True, return_validity=True
        )
        assert sums.tolist() == [11, 1.0]
        assert validity.tolist() == [True, True]

        sums = ccube([idx1, idx2]).sum(arr)
        assert arr_eq(sums, [[7.0, 4.0], [float("nan"), 1.0]])

        sums, validity = ccube([idx1, idx2]).sum(arr, return_validity=True)
        assert sums.tolist() == [[7.0, 4.0], [0.0, 1.0]]
        assert validity.tolist() == [[True, True], [False, True]]
