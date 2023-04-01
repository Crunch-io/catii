import operator
import sys
from functools import reduce

import pytest

from catii import xcube, xfuncs

from .. import arr_eq

MAXFLOAT = sys.float_info.max
arr1 = [1, 0, 1, 0, 1, 0, 1, 0]  # iindex({(1,): [0, 2, 4, 6]}, 0, (8,))
arr2 = [1, 0, 0, 1, 0, 0, 1, 0]  # iindex({(1,): [0, 3, 6]}, 0, (8,))

# The cube of arr1 x arr2 has rowids:
#      ___arr 2___
#      0       1
# a|0  [1, 4]  [3]
# r|
# r|
# 1|1  [2]     [0]


class TestXfuncMaxWorkflow:
    def test_workflow(self):
        # Other tests often simply test output; here we test intermediate results.
        f = xfuncs.xfunc_max([8, 3, 2, 9, 0, 7, 1, 5])

        cube = xcube([])
        (maxes, validity) = f.get_initial_regions(cube)
        assert arr_eq(maxes, [float("nan")])
        assert validity.tolist() == [False]
        coordinates = None
        f.fill(coordinates, (maxes, validity))
        assert maxes.tolist() == [9]
        assert validity.tolist() == [True]
        assert f.reduce(cube, (maxes, validity)) == [9]

        cube = xcube([arr1])
        (maxes, validity) = f.get_initial_regions(cube)
        assert arr_eq(maxes, [float("nan"), float("nan")])
        assert validity.tolist() == [False, False]
        coordinates = reduce(operator.add, cube.strided_dims())
        f.fill(coordinates, (maxes, validity))
        assert maxes.tolist() == [9, 8]
        assert validity.tolist() == [True, True]
        assert arr_eq(f.reduce(cube, (maxes, validity)), [9, 8])

        cube = xcube([arr1, arr2])
        (maxes, validity) = f.get_initial_regions(cube)
        assert arr_eq(
            maxes, [[float("nan"), float("nan")], [float("nan"), float("nan")]]
        )
        assert validity.tolist() == [[False, False], [False, False]]
        coordinates = reduce(operator.add, cube.strided_dims())
        f.fill(coordinates, (maxes, validity))
        assert maxes.tolist() == [[7, 9], [2, 8]]
        assert validity.tolist() == [[True, True], [True, True]]
        assert arr_eq(f.reduce(cube, (maxes, validity)), [[7, 9], [2, 8]])


class TestXfuncMaxMissingness:
    # Make sure each combination of these options works properly:
    #  * ignore_missing = True/False
    #  * return_missing_as = NaN/(<sentinel value>, False)
    #  * at least one output cell which has no inputs
    #  * fact variable (with missings) as:
    #    * single array with NaN
    #    * no NaN, but separate "validity" array
    #    * includes NaN, but separate "validity" array
    arr_with_empty_cell = [0, 0, 0, 1, 0, 0, 0, 1]

    dirty_fact = [8, 3, 2, 9, float("nan"), 7, 1, 5]
    clean_fact = [8, 3, 2, 9, MAXFLOAT, 7, 1, 5]
    fact_validity = [True, True, True, True, False, True, True, True]

    args = [
        dirty_fact,
        (clean_fact, fact_validity),
        (dirty_fact, fact_validity),
    ]

    @pytest.mark.parametrize("factvar", args)
    def test_propagate_missing_return_nan(self, factvar):
        # The cube of arr1 has rowids:
        # 0             1
        # [1, 3, 5, 7]  [0, 2, 4, 6]
        # and fact values:
        # 0             1
        # [3, 9, 7, 5]  [8, 2, nan, 1]
        maxes = xcube([arr1]).max(factvar)
        # Cell (1,) MUST be missing, because it had a missing input.
        assert arr_eq(maxes, [9, float("nan")])

        # The cube of arr1 x arr_with_empty_cell has fact values:
        #      ________arr w________
        #      0              1
        # i|0  [3, 7]         [9, 5]
        # d|
        # x|
        # 1|1  [8, 2, nan, 1] []
        maxes = xcube([arr1, self.arr_with_empty_cell]).max(factvar)
        # Cell (1, 0) MUST be missing, because it had a missing input.
        # Cell (1, 1) MUST be missing, because it had no inputs.
        assert arr_eq(maxes, [[7, 9], [float("nan"), float("nan")]])

    @pytest.mark.parametrize("factvar", args)
    def test_ignore_missing_return_nan(self, factvar):
        # The cube of arr1 has rowids:
        # 0             1
        # [1, 3, 5, 7]  [0, 2, 4, 6]
        # and fact values:
        # 0             1
        # [3, 9, 7, 5]  [8, 2, nan, 1]
        maxes = xcube([arr1]).max(factvar, ignore_missing=True)
        # Cell (1,) MUST NOT be missing, because it had a valid input.
        assert arr_eq(maxes, [9, 8])

        # The cube of arr1 x arr_with_empty_cell has fact values:
        #      ________arr w________
        #      0              1
        # i|0  [3, 7]         [9, 5]
        # d|
        # x|
        # 1|1  [8, 2, nan, 1] []
        maxes = xcube([arr1, self.arr_with_empty_cell]).max(
            factvar, ignore_missing=True
        )
        # Cell (1, 0) MUST NOT be missing, because it had a valid input.
        # Cell (1, 1) MUST be missing, because it had no inputs.
        assert arr_eq(maxes, [[7, 9], [8, float("nan")]])

    @pytest.mark.parametrize("factvar", args)
    def test_propagate_missing_return_validity(self, factvar):
        # The cube of arr1 has rowids:
        # 0             1
        # [1, 3, 5, 7]  [0, 2, 4, 6]
        # and fact values:
        # 0             1
        # [3, 9, 7, 5]  [8, 2, nan, 1]
        maxes, validity = xcube([arr1]).max(factvar, return_missing_as=(-1, False))
        # Cell (1,) MUST be missing, because it had a missing input.
        assert arr_eq(maxes, [9, -1])
        assert arr_eq(validity, [True, False])

        # The cube of arr1 x arr_with_empty_cell has fact values:
        #      ________arr w________
        #      0              1
        # i|0  [3, 7]         [9, 5]
        # d|
        # x|
        # 1|1  [8, 2, nan, 1] []
        maxes, validity = xcube([arr1, self.arr_with_empty_cell]).max(
            factvar, return_missing_as=(-1, False)
        )
        # Cell (1, 0) MUST be missing, because it had a missing input.
        # Cell (1, 1) MUST be missing, because it had no inputs.
        assert arr_eq(maxes, [[7, 9], [-1, -1]])
        assert arr_eq(validity, [[True, True], [False, False]])

    @pytest.mark.parametrize("factvar", args)
    def test_ignore_missing_return_validity(self, factvar):
        # The cube of arr1 has rowids:
        # 0             1
        # [1, 3, 5, 7]  [0, 2, 4, 6]
        # and fact values:
        # 0             1
        # [3, 9, 7, 5]  [8, 2, nan, 1]
        maxes, validity = xcube([arr1]).max(
            factvar, ignore_missing=True, return_missing_as=(-1, False)
        )
        # Cell (1,) MUST NOT be missing, because it had a valid input.
        assert arr_eq(maxes, [9, 8])
        assert arr_eq(validity, [True, True])

        # The cube of arr1 x arr_with_empty_cell has fact values:
        #      ________arr w________
        #      0              1
        # i|0  [3, 7]         [9, 5]
        # d|
        # x|
        # 1|1  [8, 2, nan, 1] []
        maxes, validity = xcube([arr1, self.arr_with_empty_cell]).max(
            factvar, ignore_missing=True, return_missing_as=(-1, False)
        )
        # Cell (1, 0) MUST NOT be missing, because it had a valid input.
        # Cell (1, 1) MUST be missing, because it had no inputs.
        assert arr_eq(maxes, [[7, 9], [8, -1]])
        assert arr_eq(validity, [[True, True], [True, False]])
