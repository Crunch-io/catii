import operator
from functools import reduce

from catii import xcube, xfuncs

from .. import arr_eq

arr1 = [1, 0, 1, 0, 0]  # iindex({(1,): [0, 2]}, 0, (5,))
arr2 = [1, 0, 0, 1, 0]  # iindex({(1,): [0, 3]}, 0, (5,))

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
        f = xfuncs.xfunc_max([8, 3, 2, 9, 0])

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
        assert maxes.tolist() == [[3, 9], [2, 8]]
        assert validity.tolist() == [[True, True], [True, True]]
        assert arr_eq(f.reduce(cube, (maxes, validity)), [[3, 9], [2, 8]])


class TestXfuncMaxIgnoreMissing:
    def test_ignore_missing(self):
        # The cube of arr1 has rowids:
        # 0           1
        # [1, 3, 4]   [0, 2]

        values = [8, 3, 2, 9, float("nan")]
        maxes = xcube([arr1]).max(values)
        # Cell (0,) MUST be missing, because it had a missing value.
        assert arr_eq(maxes, [float("nan"), 8])

        maxes = xcube([arr1]).max(values, ignore_missing=True)
        # Cell (0,) MUST NOT be missing, because it had a valid value.
        assert arr_eq(maxes, [9, 8])

        # The cube of arr1 x arr2 has values:
        #      ______arr 2_______
        #      0           1
        # a|0  [3, nan]    [9]
        # r|
        # r|
        # 1|1  [2]         [8]

        maxes = xcube([arr1, arr2]).max(values)
        # Cell (0, 0) MUST be missing, because it had a missing value.
        assert arr_eq(maxes, [[float("nan"), 9], [2, 8]])

        maxes = xcube([arr1, arr2]).max(values, ignore_missing=True)
        # Cell (0, 0) MUST NOT be missing, because it had non-missing values
        assert arr_eq(maxes, [[3, 9], [2, 8]])


class TestXfuncMaxReturnMissingAs:
    def test_return_missing_as(self):
        values = [8, 3, 2, 9, float("nan")]

        maxes = xcube([arr1]).max(values)
        assert arr_eq(maxes, [float("nan"), 8])

        maxes, validity = xcube([arr1]).max(values, return_missing_as=(0, False))
        assert arr_eq(maxes, [float("nan"), 8])
        # Cell (0,) MUST be True (valid) because it actually had inputs,
        # even though one was NaN and max() returned NaN.
        assert validity.tolist() == [True, True]

        maxes, validity = xcube([arr1]).max(
            values, ignore_missing=True, return_missing_as=(0, False)
        )
        assert maxes.tolist() == [9, 8]
        assert validity.tolist() == [True, True]

        maxes = xcube([arr1, arr2]).max(values)
        assert arr_eq(maxes, [[float("nan"), 9], [2, 8]])

        maxes, validity = xcube([arr1, arr2]).max(values, return_missing_as=(0, False))
        assert arr_eq(maxes, [[float("nan"), 9], [2, 8]])
        assert validity.tolist() == [[True, True], [True, True]]

        maxes, validity = xcube([[2, 0, 2, 0, 0]]).max(
            values, return_missing_as=(0, False)
        )
        assert arr_eq(maxes, [float("nan"), 0, 8])
        # Cell (0,) MUST be True (valid) because it actually had inputs,
        # even though one was NaN and max() returned NaN.
        # Cell (1,) MUST be False (missing) because it had no inputs.
        assert validity.tolist() == [True, False, True]
