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


class TestXfuncQuantileWorkflow:
    def _test_unweighted_workflow(self, factvar):
        # Other tests often simply test output; here we test intermediate results.
        f = xfuncs.xfunc_quantile(factvar, 0.5, ignore_missing=True)

        cube = xcube([])
        (qs,) = f.get_initial_regions(cube)
        assert arr_eq(qs, [float("nan")])
        coordinates = None
        f.fill(coordinates, (qs,))
        assert qs.tolist() == [3.0]
        assert f.reduce(cube, (qs,)) == [3.0]

        cube = xcube([arr1])
        (qs,) = f.get_initial_regions(cube)
        assert arr_eq(qs, [float("nan"), float("nan")])
        coordinates = reduce(operator.add, cube.strided_dims())
        f.fill(coordinates, (qs,))
        assert qs.tolist() == [4.0, 1.0]
        assert f.reduce(cube, (qs,)).tolist() == [4.0, 1.0]

        cube = xcube([arr1, arr2])
        (qs,) = f.get_initial_regions(cube)
        assert arr_eq(qs, [[float("nan"), float("nan")], [float("nan"), float("nan")]])
        coordinates = reduce(operator.add, cube.strided_dims())
        f.fill(coordinates, (qs,))
        assert arr_eq(qs, [[3.5, 4.0], [float("nan"), 1.0]])
        assert arr_eq(f.reduce(cube, (qs,)), [[3.5, 4.0], [float("nan"), 1.0]])

    def _test_weighted_workflow(self, factvar):
        # Other tests often simply test output; here we test intermediate results.
        f = xfuncs.xfunc_quantile(factvar, 0.5, weights=wt, ignore_missing=True)

        cube = xcube([arr1, arr2])
        (qs,) = f.get_initial_regions(cube)
        assert arr_eq(qs, [[float("nan"), float("nan")], [float("nan"), float("nan")]])
        coordinates = reduce(operator.add, cube.strided_dims())
        f.fill(coordinates, (qs,))
        assert arr_eq(qs, [[2.0, 4.0], [float("nan"), 1.0]])
        assert arr_eq(f.reduce(cube, (qs,)), [[2.0, 4.0], [float("nan"), 1.0]])

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

    def test_tuple_arr_integer_workflow(self):
        factvar = ([1, 2, 3, 4, 5], [True, True, False, True, True])
        self._test_unweighted_workflow(factvar)
        self._test_weighted_workflow(factvar)


class TestXfuncQuantileWeights:
    def test_weights(self):
        factvar = [1.0, 2.0, 3.0, 4.0, 5.0]

        qs = xcube([arr1]).quantile(factvar, 0.5, weights=None)
        assert arr_eq(qs, [4.0, 2.0])

        qs = xcube([arr1]).quantile(factvar, 0.5, weights=[0.25, 0.3, 0.99, 1.0, 0.5])
        assert arr_eq(qs, [3.2, 1.74747475])

        qs = xcube([arr1]).quantile(
            factvar, 0.5, weights=[0.25, 0.3, 0.99, 1.0, float("nan")]
        )
        assert arr_eq(qs, [float("nan"), 1.74747475])

        qs = xcube([arr1]).quantile(
            factvar,
            0.5,
            weights=[0.25, 0.3, 0.99, 1.0, float("nan")],
            ignore_missing=True,
        )
        assert arr_eq(qs, [2.7, 1.74747475])

        qs = xcube([arr1]).quantile(factvar, 0.5, weights=0.5)
        assert arr_eq(qs, [3.0, 1.0])

        qs = xcube([arr1]).quantile(factvar, 0.5, weights=float("nan"))
        assert arr_eq(qs, [float("nan"), float("nan")])

        qs = xcube([arr1]).quantile(
            factvar, 0.5, weights=float("nan"), ignore_missing=True
        )
        assert arr_eq(qs, [float("nan"), float("nan")])


class TestXfuncQuantileIgnoreMissing:
    def test_ignore_missing(self):
        factvar = [1.0, 2.0, float("nan"), 4.0, 5.0]

        # The cube of arr1 has rowids:
        # 0           1
        # [1, 3, 4]  [0, 2]
        # and factvar values:
        # 0           1
        # [2.0, 4.0, 5.0]  [1.0, nan]

        qs = xcube([arr1]).quantile(factvar, 0.5)
        # Cell (1,) MUST be missing, because it had a missing input.
        assert arr_eq(qs, [4.0, float("nan")])

        qs = xcube([arr1]).quantile(factvar, 0.5, ignore_missing=True)
        # Cell (1,) MUST NOT be missing, because it had a valid input.
        assert arr_eq(qs, [4.0, 1.0])

        # The cube of arr1 x arr2 has factvar values:
        #      ______arr 2______
        #      0           1
        # a|0  [2.0, 5.0]  [4.0]
        # r|
        # r|
        # 1|1  [nan]       [1.0]

        qs = xcube([arr1, arr2]).quantile(factvar, 0.5)
        # Cell (1, 0) MUST be missing, because it had a missing factvar value.
        assert arr_eq(qs, [[3.5, 4.0], [float("nan"), 1.0]])

        qs = xcube([arr1, arr2]).quantile(factvar, 0.5, ignore_missing=True)
        # Cell (1, 0) MUST be missing, because it had no non-missing inputs.
        assert arr_eq(qs, [[3.5, 4.0], [float("nan"), 1.0]])


class TestXfuncQuantileReturnMissingAs:
    def test_return_missing_as(self):
        factvar = [1.0, 2.0, float("nan"), 4.0, 5.0]

        qs = xcube([arr1]).quantile(factvar, 0.5)
        assert arr_eq(qs, [4.0, float("nan")])

        qs, validity = xcube([arr1]).quantile(
            factvar, 0.5, return_missing_as=(0, False)
        )
        assert qs.tolist() == [4.0, 0.0]
        assert validity.tolist() == [True, False]

        qs, validity = xcube([arr1]).quantile(
            factvar, 0.5, ignore_missing=True, return_missing_as=(0, False)
        )
        assert qs.tolist() == [4.0, 1.0]
        assert validity.tolist() == [True, True]

        qs = (xcube([arr1, arr2]).quantile(factvar, 0.5),)
        assert arr_eq(qs, [[3.5, 4.0], [float("nan"), 1.0]])

        qs, validity = xcube([arr1, arr2]).quantile(
            factvar, 0.5, return_missing_as=(0, False)
        )
        assert qs.tolist() == [[3.5, 4.0], [0.0, 1.0]]
        assert validity.tolist() == [[True, True], [False, True]]
