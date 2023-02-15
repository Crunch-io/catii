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


class TestXfuncValidCountWorkflow:
    def _test_unweighted_workflow(self, factvar):
        # Other tests often simply test output; here we test intermediate results.
        f = xfuncs.xfunc_valid_count(factvar)

        cube = xcube([])
        counts, missings = f.get_initial_regions(cube)
        assert counts.tolist() == [0]
        assert missings.tolist() == [0]
        coordinates = None
        f.fill(coordinates, (counts, missings))
        assert counts.tolist() == [4]
        assert missings.tolist() == [1]
        assert numpy.isnan(f.reduce(cube, (counts, missings)))

        cube = xcube([arr1])
        counts, missings = f.get_initial_regions(cube)
        assert counts.tolist() == [0, 0]
        assert missings.tolist() == [0, 0]
        coordinates = reduce(operator.add, cube.strided_dims())
        f.fill(coordinates, (counts, missings))
        assert counts.tolist() == [3, 1]
        assert missings.tolist() == [0, 1]
        assert arr_eq(f.reduce(cube, (counts, missings)), [3, float("nan")])

        cube = xcube([arr1, arr2])
        counts, missings = f.get_initial_regions(cube)
        assert counts.tolist() == [[0, 0], [0, 0]]
        assert missings.tolist() == [[0, 0], [0, 0]]
        coordinates = reduce(operator.add, cube.strided_dims())
        f.fill(coordinates, (counts, missings))
        assert counts.tolist() == [[2, 1], [0, 1]]
        assert missings.tolist() == [[0, 0], [1, 0]]
        assert arr_eq(f.reduce(cube, (counts, missings)), [[2, 1], [float("nan"), 1]])

    def _test_weighted_workflow(self, factvar):
        # Other tests often simply test output; here we test intermediate results.
        f = xfuncs.xfunc_valid_count(factvar, weights=wt)

        cube = xcube([arr1, arr2])
        counts, missings = f.get_initial_regions(cube)
        assert counts.tolist() == [[0.0, 0.0], [0.0, 0.0]]
        assert missings.tolist() == [[0, 0], [0, 0]]
        coordinates = reduce(operator.add, cube.strided_dims())
        f.fill(coordinates, (counts, missings))
        assert counts.tolist() == [[0.3, 1.0], [0.0, 0.25]]
        assert missings.tolist() == [[1, 0], [1, 0]]
        assert arr_eq(
            f.reduce(cube, (counts, missings)),
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


class TestXfuncValidCountIgnoreMissing:
    def test_ignore_missing(self):
        factvar = [1.0, 2.0, float("nan"), 4.0, 5.0]

        # The cube of arr1 has rowids:
        # 0           1
        # [1, 3, 4]  [0, 2]
        # and factvar values:
        # 0           1
        # [2.0, 4.0, 5.0]  [1.0, nan]

        counts = xcube([arr1]).valid_count(factvar)
        # Cell (1,) MUST be missing, because it had a missing input.
        assert arr_eq(counts, [3, float("nan")])

        counts = xcube([arr1]).valid_count(factvar, ignore_missing=True)
        # Cell (1,) MUST NOT be missing, because it had a valid input.
        assert arr_eq(counts, [3, 1])

        # The cube of arr1 x arr2 has factvar values:
        #      ______idx 2______
        #      0           1
        # i|0  [2.0, 5.0]  [4.0]
        # d|
        # x|
        # 1|1  [nan]       [1.0]

        counts = xcube([arr1, arr2]).valid_count(factvar)
        # Cell (1, 0) MUST be missing, because it had a missing arr value.
        assert arr_eq(counts, [[2, 1], [float("nan"), 1]])

        counts = xcube([arr1, arr2]).valid_count(factvar, ignore_missing=True)
        # Cell (1, 0) MUST be missing, because it had no non-missing inputs.
        assert arr_eq(counts, [[2, 1], [float("nan"), 1]])


class TestXfuncValidCountReturnValidity:
    def test_return_validity(self):
        factvar = [1.0, 2.0, float("nan"), 4.0, 5.0]

        counts = xcube([arr1]).valid_count(factvar)
        assert arr_eq(counts, [3, float("nan")])

        counts, validity = xcube([arr1]).valid_count(factvar, return_validity=True)
        assert counts.tolist() == [3, 0]
        assert validity.tolist() == [True, False]

        counts, validity = xcube([arr1]).valid_count(
            factvar, ignore_missing=True, return_validity=True
        )
        assert counts.tolist() == [3, 1]
        assert validity.tolist() == [True, True]

        counts = xcube([arr1, arr2]).valid_count(factvar)
        assert arr_eq(counts, [[2, 1], [float("nan"), 1]])

        counts, validity = xcube([arr1, arr2]).valid_count(
            factvar, return_validity=True
        )
        assert counts.tolist() == [[2, 1], [0, 1]]
        assert validity.tolist() == [[True, True], [False, True]]
