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


class TestFfuncCountWorkflow:
    def _test_unweighted_workflow(self):
        # Other tests often simply test output; here we test intermediate results.
        f = ffuncs.ffunc_count(N=5)

        cube = ccube([])
        (counts,) = f.get_initial_regions(cube)
        assert counts.tolist() == 5
        f.fill(cube, (counts,))
        assert counts.tolist() == 5
        assert f.reduce(cube, (counts,)) == 5

        cube = ccube([idx1])
        (counts,) = f.get_initial_regions(cube)
        assert counts.tolist() == [0, 0, 5]
        f.fill(cube, (counts,))
        assert counts.tolist() == [0, 2, 5]
        assert arr_eq(f.reduce(cube, (counts,)), [3, 2])

        cube = ccube([idx1, idx2])
        (counts,) = f.get_initial_regions(cube)
        assert counts.tolist() == [[0, 0, 0], [0, 0, 0], [0, 0, 5]]
        f.fill(cube, (counts,))
        assert counts.tolist() == [[0, 0, 0], [0, 1, 2], [0, 2, 5]]
        assert arr_eq(f.reduce(cube, (counts,)), [[2, 1], [1, 1]])

    def _test_weighted_workflow(self):
        # Other tests often simply test output; here we test intermediate results.
        f = ffuncs.ffunc_count(weights=wt)
        weight, validity = ffuncs.as_separate_validity(wt)

        cube = ccube([idx1, idx2])
        counts, missings = f.get_initial_regions(cube)
        assert counts.tolist() == [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, wt[validity].sum()],
        ]
        assert missings.tolist() == [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 1],
        ]
        f.fill(cube, (counts, missings))
        assert counts.tolist() == [
            [0.0, 0.0, 0.0],
            [0.0, 0.25, (0.25 + 0.99)],
            [0.0, (0.25 + 1.0), wt[validity].sum()],
        ]
        assert missings.tolist() == [
            [0, 0, 0],  # idx1 == 0
            [0, 0, 0],  # idx1 == 1
            [0, 0, 1],  # idx1 == any
        ]
        assert arr_eq(
            f.reduce(cube, (counts, missings)), [[float("nan"), 1.0], [0.99, 0.25]],
        )

    def test_single_arr_with_nan_workflow(self):
        self._test_unweighted_workflow()
        self._test_weighted_workflow()


class TestFfuncCountWeights:
    def test_weights(self):
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


class TestFfuncCountIgnoreMissing:
    def test_ignore_missing(self):
        # The cube of idx1 has rowids:
        # 0           1
        # [1, 3, 4]   [0, 2]
        # and weights:
        # 0                1
        # [0.3, 1.0, nan]  [0.25, 0.99]

        counts = ccube([idx1]).count(weights=wt)
        # Cell (0,) MUST be missing, because it had a missing weight.
        assert arr_eq(counts, [float("nan"), (0.25 + 0.99)])

        counts = ccube([idx1]).count(weights=wt, ignore_missing=True)
        # Cell (0,) MUST NOT be missing, because it had a valid weight.
        assert arr_eq(counts, [(0.3 + 1.0), (0.25 + 0.99)])

        # The cube of idx1 x idx2 has weights:
        #      ______idx 2_______
        #      0           1
        # i|0  [0.3, nan]  [1.0]
        # d|
        # x|
        # 1|1  [0.99       [0.25]

        counts = ccube([idx1, idx2]).count(weights=wt)
        # Cell (0, 0) MUST be missing, because it had a missing weight.
        assert arr_eq(counts, [[float("nan"), 1.0], [0.99, 0.25]])

        counts = ccube([idx1, idx2]).count(weights=wt, ignore_missing=True)
        # Cell (0, 0) MUST NOT be missing, because it had non-missing weights
        assert arr_eq(counts, [[0.3, 1.0], [0.99, 0.25]])


class TestFfuncCountReturnValidity:
    def test_return_validity(self):
        counts = ccube([idx1]).count(weights=wt)
        assert arr_eq(counts, [float("nan"), (0.25 + 0.99)])

        counts, validity = ccube([idx1]).count(weights=wt, return_validity=True)
        assert counts.tolist() == [0, (0.25 + 0.99)]
        assert validity.tolist() == [False, True]

        counts, validity = ccube([idx1]).count(
            weights=wt, ignore_missing=True, return_validity=True
        )
        assert counts.tolist() == [(0.3 + 1.0), (0.25 + 0.99)]
        assert validity.tolist() == [True, True]

        counts = ccube([idx1, idx2]).count(weights=wt)
        assert arr_eq(counts, [[float("nan"), 1.0], [0.99, 0.25]])

        counts, validity = ccube([idx1, idx2]).count(weights=wt, return_validity=True)
        assert counts.tolist() == [[0, 1.0], [0.99, 0.25]]
        assert validity.tolist() == [[False, True], [True, True]]
