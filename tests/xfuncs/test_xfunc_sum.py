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


class TestXfuncSumWorkflow:
    def _test_unweighted_workflow(self, factvar):
        # Other tests often simply test output; here we test intermediate results.
        f = xfuncs.xfunc_sum(factvar)

        cube = xcube([])
        sums, missings = f.get_initial_regions(cube)
        assert sums.tolist() == [0.0]
        assert missings.tolist() == [0]
        coordinates = None
        f.fill(coordinates, (sums, missings))
        assert sums.tolist() == [12.0]
        assert missings.tolist() == [1]
        assert numpy.isnan(f.reduce(cube, (sums, missings)))

        cube = xcube([arr1])
        sums, missings = f.get_initial_regions(cube)
        assert sums.tolist() == [0.0, 0.0]
        assert missings.tolist() == [0, 0]
        coordinates = reduce(operator.add, cube.strided_dims())
        f.fill(coordinates, (sums, missings))
        assert sums.tolist() == [11.0, 1.0]
        assert missings.tolist() == [0, 1]
        assert arr_eq(f.reduce(cube, (sums, missings)), [11, float("nan")])

        cube = xcube([arr1, arr2])
        sums, missings = f.get_initial_regions(cube)
        assert sums.tolist() == [[0.0, 0.0], [0.0, 0.0]]
        assert missings.tolist() == [[0, 0], [0, 0]]
        coordinates = reduce(operator.add, cube.strided_dims())
        f.fill(coordinates, (sums, missings))
        assert sums.tolist() == [[7.0, 4.0], [0.0, 1.0]]
        assert missings.tolist() == [[0, 0], [1, 0]]
        assert arr_eq(
            f.reduce(cube, (sums, missings)), [[7.0, 4.0], [float("nan"), 1.0]]
        )

    def _test_weighted_workflow(self, factvar):
        # Other tests often simply test output; here we test intermediate results.
        f = xfuncs.xfunc_sum(factvar, weights=wt)

        cube = xcube([arr1, arr2])
        sums, missings = f.get_initial_regions(cube)
        assert sums.tolist() == [[0.0, 0.0], [0.0, 0.0]]
        assert missings.tolist() == [[0, 0], [0, 0]]
        coordinates = reduce(operator.add, cube.strided_dims())
        f.fill(coordinates, (sums, missings))
        assert sums.tolist() == [[0.6, 4.0], [0.0, 0.25]]
        assert missings.tolist() == [[1, 0], [1, 0]]
        assert arr_eq(
            f.reduce(cube, (sums, missings)),
            [[float("nan"), 4.0], [float("nan"), 0.25]],
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


class TestXfuncSumWeights:
    def test_weights(self):
        factvar = [1.0, 2.0, 3.0, 4.0, 5.0]

        sums = xcube([arr1]).sum(factvar, weights=None)
        assert arr_eq(sums, [11.0, 4.0])

        sums = xcube([arr1]).sum(factvar, weights=[0.25, 0.3, 0.99, 1.0, 0.5])
        assert arr_eq(sums, [7.1, 3.22])

        sums = xcube([arr1]).sum(factvar, weights=[0.25, 0.3, 0.99, 1.0, float("nan")])
        assert arr_eq(sums, [float("nan"), 3.22])

        sums = xcube([arr1]).sum(
            factvar, weights=[0.25, 0.3, 0.99, 1.0, float("nan")], ignore_missing=True
        )
        assert arr_eq(sums, [4.6, 3.22])

        sums = xcube([arr1]).sum(factvar, weights=0.5)
        assert arr_eq(sums, [5.5, 2.0])

        sums = xcube([arr1]).sum(factvar, weights=float("nan"))
        assert arr_eq(sums, [float("nan"), float("nan")])

        sums = xcube([arr1]).sum(factvar, weights=float("nan"), ignore_missing=True)
        assert arr_eq(sums, [float("nan"), float("nan")])


class TestXfuncSumIgnoreMissing:
    def test_ignore_missing(self):
        factvar = [1.0, 2.0, float("nan"), 4.0, 5.0]

        # The cube of arr1 has rowids:
        # 0           1
        # [1, 3, 4]  [0, 2]
        # and factvar values:
        # 0           1
        # [2.0, 4.0, 5.0]  [1.0, nan]

        sums = xcube([arr1]).sum(factvar)
        # Cell (1,) MUST be missing, because it had a missing input.
        assert arr_eq(sums, [11.0, float("nan")])

        sums = xcube([arr1]).sum(factvar, ignore_missing=True)
        # Cell (1,) MUST NOT be missing, because it had a valid input.
        assert arr_eq(sums, [11.0, 1.0])

        # The cube of arr1 x arr2 has factvar values:
        #      ______arr 2______
        #      0           1
        # a|0  [2.0, 5.0]  [4.0]
        # r|
        # r|
        # 1|1  [nan]       [1.0]

        sums = xcube([arr1, arr2]).sum(factvar)
        # Cell (1, 0) MUST be missing, because it had a missing factvar value.
        assert arr_eq(sums, [[7.0, 4.0], [float("nan"), 1.0]])

        sums = xcube([arr1, arr2]).sum(factvar, ignore_missing=True)
        # Cell (1, 0) MUST be missing, because it had no non-missing inputs.
        assert arr_eq(sums, [[7.0, 4.0], [float("nan"), 1.0]])


class TestXfuncSumReturnValidity:
    def test_return_validity(self):
        factvar = [1.0, 2.0, float("nan"), 4.0, 5.0]

        sums = xcube([arr1]).sum(factvar)
        assert arr_eq(sums, [11, float("nan")])

        sums, validity = xcube([arr1]).sum(factvar, return_validity=True)
        assert sums.tolist() == [11, 0.0]
        assert validity.tolist() == [True, False]

        sums, validity = xcube([arr1]).sum(
            factvar, ignore_missing=True, return_validity=True
        )
        assert sums.tolist() == [11, 1.0]
        assert validity.tolist() == [True, True]

        sums = xcube([arr1, arr2]).sum(factvar)
        assert arr_eq(sums, [[7.0, 4.0], [float("nan"), 1.0]])

        sums, validity = xcube([arr1, arr2]).sum(factvar, return_validity=True)
        assert sums.tolist() == [[7.0, 4.0], [0.0, 1.0]]
        assert validity.tolist() == [[True, True], [False, True]]
