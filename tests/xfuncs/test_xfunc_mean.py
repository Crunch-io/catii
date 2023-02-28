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


class TestXfuncMeanWorkflow:
    def _test_unweighted_workflow(self, factvar):
        # Other tests often simply test output; here we test intermediate results.
        f = xfuncs.xfunc_mean(factvar)

        cube = xcube([])
        sums, counts, missings = f.get_initial_regions(cube)
        assert sums.tolist() == [0.0]
        assert counts.tolist() == [0]
        assert missings.tolist() == [0]
        coordinates = None
        f.fill(coordinates, (sums, counts, missings))
        assert sums.tolist() == [12.0]
        assert counts.tolist() == [4]
        assert missings.tolist() == [1]
        assert numpy.isnan(f.reduce(cube, (sums, counts, missings)))

        cube = xcube([arr1])
        sums, counts, missings = f.get_initial_regions(cube)
        assert sums.tolist() == [0.0, 0.0]
        assert counts.tolist() == [0, 0]
        assert missings.tolist() == [0, 0]
        coordinates = reduce(operator.add, cube.strided_dims())
        f.fill(coordinates, (sums, counts, missings))
        assert sums.tolist() == [11.0, 1.0]
        assert counts.tolist() == [3, 1]
        assert missings.tolist() == [0, 1]
        assert arr_eq(
            f.reduce(cube, (sums, counts, missings)), [11 / 3.0, float("nan")]
        )

        cube = xcube([arr1, arr2])
        sums, counts, missings = f.get_initial_regions(cube)
        assert sums.tolist() == [[0.0, 0.0], [0.0, 0.0]]
        assert counts.tolist() == [[0.0, 0.0], [0.0, 0.0]]
        assert missings.tolist() == [[0.0, 0.0], [0.0, 0.0]]
        coordinates = reduce(operator.add, cube.strided_dims())
        f.fill(coordinates, (sums, counts, missings))
        assert sums.tolist() == [[7.0, 4.0], [0, 1.0]]
        assert counts.tolist() == [[2, 1], [0, 1]]
        assert missings.tolist() == [[0, 0], [1, 0]]
        assert arr_eq(
            f.reduce(cube, (sums, counts, missings)), [[3.5, 4.0], [float("nan"), 1.0]]
        )

    def _test_weighted_workflow(self, factvar):
        # Other tests often simply test output; here we test intermediate results.
        f = xfuncs.xfunc_mean(factvar, weights=wt)

        cube = xcube([arr1, arr2])
        sums, counts, missings = f.get_initial_regions(cube)
        assert sums.tolist() == [[0.0, 0.0], [0.0, 0.0]]
        assert counts.tolist() == [[0, 0], [0, 0]]
        assert missings.tolist() == [[0, 0], [0, 0]]
        coordinates = reduce(operator.add, cube.strided_dims())
        f.fill(coordinates, (sums, counts, missings))
        assert sums.tolist() == [[0.6, 4.0], [0.0, 0.25]]
        assert counts.tolist() == [[0.3, 1.0], [0.0, 0.25]]
        assert missings.tolist() == [[1, 0], [1, 0]]
        assert arr_eq(
            f.reduce(cube, (sums, counts, missings)),
            [[float("nan"), 4.0], [float("nan"), 1.0]],
        )

    def test_single_arr_with_nan_workflow(self):
        factvar = [1.0, 2.0, float("nan"), 4.0, 5.0]
        self._test_unweighted_workflow(factvar)
        self._test_weighted_workflow(factvar)

    def test_tuple_factvar_workflow(self):
        factvar = ([1.0, 2.0, 3.0, 4.0, 5.0], [True, True, False, True, True])
        self._test_unweighted_workflow(factvar)
        self._test_weighted_workflow(factvar)

    def test_tuple_factvar_with_nan_workflow(self):
        factvar = ([1.0, 2.0, float("nan"), 4.0, 5.0], [True, True, False, True, True])
        self._test_unweighted_workflow(factvar)
        self._test_weighted_workflow(factvar)


class TestXfuncMeanIgnoreMissing:
    def test_ignore_missing(self):
        factvar = [1.0, 2.0, float("nan"), 4.0, 5.0]

        # The cube of arr1 has rowids:
        # 0           1
        # [1, 3, 4]  [0, 2]
        # and fact values:
        # 0           1
        # [2.0, 4.0, 5.0]  [1.0, nan]

        means = xcube([arr1]).mean(factvar)
        # Cell (1,) MUST be missing, because it had a missing input.
        assert arr_eq(means, [11 / 3.0, float("nan")])

        means = xcube([arr1]).mean(factvar, ignore_missing=True)
        # Cell (1,) MUST NOT be missing, because it had a valid input.
        assert arr_eq(means, [11 / 3.0, 1.0])

        # The cube of arr1 x arr2 has fact values:
        #      ______arr 2______
        #      0           1
        # a|0  [2.0, 5.0]  [4.0]
        # r|
        # r|
        # 1|1  [nan]       [1.0]

        means = xcube([arr1, arr2]).mean(factvar)
        # Cell (1, 0) MUST be missing, because it had a missing fact value.
        assert arr_eq(means, [[3.5, 4.0], [float("nan"), 1.0]])

        means = xcube([arr1, arr2]).mean(factvar, ignore_missing=True)
        # Cell (1, 0) MUST be missing, because it had no non-missing inputs.
        assert arr_eq(means, [[3.5, 4.0], [float("nan"), 1.0]])


class TestXfuncMeanReturnMissingAs:
    def test_return_missing_as(self):
        factvar = [1.0, 2.0, float("nan"), 4.0, 5.0]

        means = xcube([arr1]).mean(factvar)
        assert arr_eq(means, [11 / 3.0, float("nan")])

        means, validity = xcube([arr1]).mean(factvar, return_missing_as=(0, False))
        assert means.tolist() == [11 / 3.0, 0.0]
        assert validity.tolist() == [True, False]

        means, validity = xcube([arr1]).mean(
            factvar, ignore_missing=True, return_missing_as=(0, False)
        )
        assert means.tolist() == [11 / 3.0, 1.0]
        assert validity.tolist() == [True, True]

        means = xcube([arr1, arr2]).mean(factvar)
        assert arr_eq(means, [[3.5, 4.0], [float("nan"), 1.0]])

        means, validity = xcube([arr1, arr2]).mean(
            factvar, return_missing_as=(0, False)
        )
        assert means.tolist() == [[3.5, 4.0], [0.0, 1.0]]
        assert validity.tolist() == [[True, True], [False, True]]
