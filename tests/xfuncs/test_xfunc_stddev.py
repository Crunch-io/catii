import operator
from functools import reduce

import numpy

from catii import xcube, xfuncs

from .. import arr_eq

arr1 = [1, 0, 1, 0, 1, 0, 1, 0]  # iindex({(1,): [0, 2, 4, 6]}, 0, (8,))
arr2 = [1, 0, 0, 1, 0, 0, 1, 0]  # iindex({(1,): [0, 3, 6]}, 0, (8,))
wt = numpy.array([0.25, 0.3, 0.99, 1.0, float("nan"), 0.5, 0.75, 0.0])

# The cube of arr1 x arr2 has rowids:
#      ___arr 2___
#      0       1
# a|0  [1, 5, 7]  [3]
# r|
# r|
# 1|1  [2, 4]     [0, 6]
#
# and therefore weights:
#      ________arr 2_________
#      0                1
# a|0  [0.3, 0.5, 0.0]  [1.0]
# r|
# r|
# 1|1  [0.99, nan]      [0.25, 0.75]


class TestXfuncStddevWorkflow:
    def _test_unweighted_workflow(self, factvar):
        # Other tests often simply test output; here we test intermediate results.
        f = xfuncs.xfunc_stddev(factvar)

        cube = xcube([])
        stddevs, missings = f.get_initial_regions(cube)
        assert arr_eq(stddevs, [float("nan")])
        assert missings.tolist() == [0]
        coordinates = None
        f.fill(coordinates, (stddevs, missings))
        assert stddevs.tolist() == [36.2832770085089]
        assert missings.tolist() == [1]
        assert numpy.isnan(f.reduce(cube, (stddevs, missings)))

        cube = xcube([arr1])
        stddevs, missings = f.get_initial_regions(cube)
        assert arr_eq(stddevs, [float("nan"), float("nan")])
        assert missings.tolist() == [0, 0]
        coordinates = reduce(operator.add, cube.strided_dims())
        f.fill(coordinates, (stddevs, missings))
        assert arr_eq(stddevs, [48.027769744874334, float("nan")])
        assert missings.tolist() == [0, 1]
        assert arr_eq(
            f.reduce(cube, (stddevs, missings)), [48.027769744874334, float("nan")]
        )

        cube = xcube([arr1, arr2])
        stddevs, missings = f.get_initial_regions(cube)
        assert arr_eq(
            stddevs, [[float("nan"), float("nan")], [float("nan"), float("nan")]]
        )
        assert missings.tolist() == [[0, 0], [0, 0]]
        coordinates = reduce(operator.add, cube.strided_dims())
        f.fill(coordinates, (stddevs, missings))
        assert arr_eq(
            stddevs,
            [[55.46169849544819, float("nan")], [float("nan"), 4.242640687119285]],
        )
        assert missings.tolist() == [[0, 0], [1, 0]]
        assert arr_eq(
            f.reduce(cube, (stddevs, missings)),
            [[55.46169849544819, float("nan")], [float("nan"), 4.242640687119285]],
        )

    def _test_weighted_workflow(self, factvar):
        # Other tests often simply test output; here we test intermediate results.
        f = xfuncs.xfunc_stddev(factvar, weights=wt)

        cube = xcube([arr1, arr2])
        stddevs, missings = f.get_initial_regions(cube)
        assert arr_eq(
            stddevs, [[float("nan"), float("nan")], [float("nan"), float("nan")]]
        )
        assert missings.tolist() == [[0, 0], [0, 0]]
        coordinates = reduce(operator.add, cube.strided_dims())
        f.fill(coordinates, (stddevs, missings))
        assert arr_eq(
            stddevs, [[2.37170825, float("nan")], [float("nan"), 3.67423461]],
        )
        assert missings.tolist() == [[0, 0], [2, 0]]
        assert arr_eq(
            f.reduce(cube, (stddevs, missings)),
            [[2.37170825, float("nan")], [float("nan"), 3.67423461]],
        )

    def test_single_arr_with_nan_workflow(self):
        factvar = [1.0, 2.0, float("nan"), 4.0, 5.0, 6.0, 7.0, 100.0]
        self._test_unweighted_workflow(factvar)
        self._test_weighted_workflow(factvar)

    def test_tuple_factvar_workflow(self):
        factvar = (
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 100.0],
            [True, True, False, True, True, True, True, True],
        )
        self._test_unweighted_workflow(factvar)
        self._test_weighted_workflow(factvar)

    def test_tuple_factvar_with_nan_workflow(self):
        factvar = (
            [1.0, 2.0, float("nan"), 4.0, 5.0, 6.0, 7.0, 100.0],
            [True, True, False, True, True, True, True, True],
        )
        self._test_unweighted_workflow(factvar)
        self._test_weighted_workflow(factvar)


class TestXfuncStddevIgnoreMissing:
    def test_ignore_missing(self):
        factvar = [1.0, 2.0, float("nan"), 4.0, 5.0, 6.0, 7.0, 100.0]

        # The cube of arr1 has rowids:
        # 0           1
        # [1, 3, 5, 7]  [0, 2, 4, 6]
        # and fact values:
        # 0           1
        # [2.0, 4.0, 6.0, 100.0]  [1.0, nan, 5.0, 7.0]

        stddevs = xcube([arr1]).stddev(factvar)
        # Cell (1,) MUST be missing, because it had a missing input.
        assert arr_eq(stddevs, [48.02776974, float("nan")])

        stddevs = xcube([arr1]).stddev(factvar, ignore_missing=True)
        # Cell (1,) MUST NOT be missing, because it had a valid input.
        assert arr_eq(stddevs, [48.02776974, 3.055050463303893])

        # The cube of arr1 x arr2 has fact values:
        #      ________arr 2________
        #      0                  1
        # a|0  [2.0, 6.0, 100.0]  [4.0]
        # r|
        # r|
        # 1|1  [nan, 5.0]     [1.0, 7.0]

        stddevs = xcube([arr1, arr2]).stddev(factvar)
        assert arr_eq(
            stddevs,
            [
                # Cell (0, 1) MUST be missing, because although it had
                # a non-missing input, it was only one and cannot have
                # a standard deviation.
                [55.46169849544819, float("nan")],
                # Cell (1, 0) MUST be missing, because it had a missing fact.
                [float("nan"), 4.24264069],
            ],
        )

        stddevs = xcube([arr1, arr2]).stddev(factvar, ignore_missing=True)
        assert arr_eq(
            stddevs,
            [
                # Cell (0, 1) MUST be missing, because although it had
                # a non-missing input, it was only one and cannot have
                # a standard deviation.
                [55.46169849544819, float("nan")],
                # Cell (1, 0) MUST be missing, because although it had
                # a non-missing input, it was only one and cannot have
                # a standard deviation.
                [float("nan"), 4.24264069],
            ],
        )


class TestXfuncStddevReturnMissingAs:
    def test_return_missing_as(self):
        factvar = [1.0, 2.0, float("nan"), 4.0, 5.0, 6.0, 7.0, 100.0]

        stddevs = xcube([arr1]).stddev(factvar)
        assert arr_eq(stddevs, [48.02776974, float("nan")])

        stddevs, validity = xcube([arr1]).stddev(factvar, return_missing_as=(0, False))
        assert stddevs.tolist() == [48.027769744874334, 0.0]
        assert validity.tolist() == [True, False]

        stddevs, validity = xcube([arr1]).stddev(
            factvar, ignore_missing=True, return_missing_as=(0, False)
        )
        assert stddevs.tolist() == [48.027769744874334, 3.055050463303893]
        assert validity.tolist() == [True, True]

        stddevs = xcube([arr1, arr2]).stddev(factvar)
        assert arr_eq(
            stddevs,
            [[55.46169849544819, float("nan")], [float("nan"), 4.242640687119285]],
        )

        stddevs, validity = xcube([arr1, arr2]).stddev(
            factvar, return_missing_as=(0, False)
        )
        assert arr_eq(
            stddevs, [[55.46169849544819, float("nan")], [0.0, 4.242640687119285]]
        )
        assert validity.tolist() == [[True, True], [False, True]]
