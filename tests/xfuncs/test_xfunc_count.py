import operator
from functools import reduce

import numpy

from catii import xcube, xfuncs

from .. import arr_eq

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
        counts, missings = f.get_initial_regions(cube)
        assert counts.tolist() == [[0.0, 0.0], [0.0, 0.0]]
        assert missings.tolist() == [[0, 0], [0, 0]]
        coordinates = reduce(operator.add, cube.strided_dims())
        f.fill(coordinates, (counts, missings))
        assert counts.tolist() == [[0.3, 1.0], [0.99, 0.25]]
        assert missings.tolist() == [[1, 0], [0, 0]]
        assert arr_eq(
            f.reduce(cube, (counts, missings)), [[float("nan"), 1.0], [0.99, 0.25]],
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


class TestXfuncCountIgnoreMissing:
    def test_ignore_missing(self):
        # The cube of arr1 has rowids:
        # 0           1
        # [1, 3, 4]   [0, 2]
        # and weights:
        # 0                1
        # [0.3, 1.0, nan]  [0.25, 0.99]

        counts = xcube([arr1]).count(weights=wt)
        # Cell (0,) MUST be missing, because it had a missing weight.
        assert arr_eq(counts, [float("nan"), (0.25 + 0.99)])

        counts = xcube([arr1]).count(weights=wt, ignore_missing=True)
        # Cell (0,) MUST NOT be missing, because it had a valid weight.
        assert arr_eq(counts, [(0.3 + 1.0), (0.25 + 0.99)])

        # The cube of arr1 x arr2 has weights:
        #      ______idx 2_______
        #      0           1
        # i|0  [0.3, nan]  [1.0]
        # d|
        # x|
        # 1|1  [0.99       [0.25]

        counts = xcube([arr1, arr2]).count(weights=wt)
        # Cell (0, 0) MUST be missing, because it had a missing weight.
        assert arr_eq(counts, [[float("nan"), 1.0], [0.99, 0.25]])

        counts = xcube([arr1, arr2]).count(weights=wt, ignore_missing=True)
        # Cell (0, 0) MUST NOT be missing, because it had non-missing weights
        assert arr_eq(counts, [[0.3, 1.0], [0.99, 0.25]])


class TestXfuncCountReturnValidity:
    def test_return_validity(self):
        counts = xcube([arr1]).count(weights=wt)
        assert arr_eq(counts, [float("nan"), (0.25 + 0.99)])

        counts, validity = xcube([arr1]).count(weights=wt, return_validity=True)
        assert counts.tolist() == [0, (0.25 + 0.99)]
        assert validity.tolist() == [False, True]

        counts, validity = xcube([arr1]).count(
            weights=wt, ignore_missing=True, return_validity=True
        )
        assert counts.tolist() == [(0.3 + 1.0), (0.25 + 0.99)]
        assert validity.tolist() == [True, True]

        counts = xcube([arr1, arr2]).count(weights=wt)
        assert arr_eq(counts, [[float("nan"), 1.0], [0.99, 0.25]])

        counts, validity = xcube([arr1, arr2]).count(weights=wt, return_validity=True)
        assert counts.tolist() == [[0, 1.0], [0.99, 0.25]]
        assert validity.tolist() == [[False, True], [True, True]]
