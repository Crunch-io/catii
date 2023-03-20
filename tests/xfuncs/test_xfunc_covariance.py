import operator
from functools import reduce

import numpy

from catii import xcube, xfuncs

from .. import arr_eq

NaN = xfuncs.NaN

arr1 = [1, 0, 1, 0, 1, 0, 1, 0]  # iindex({(1,): [0, 2, 4, 6]}, 0, (8,))
arr2 = [1, 0, 0, 1, 0, 0, 1, 0]  # iindex({(1,): [0, 3, 6]}, 0, (8,))
wt = numpy.array([0.25, 0.3, 0.99, 1.0, NaN, 0.5, 0.75, 0.0])

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


class TestXfuncCovarianceWorkflow:
    def _test_unweighted_workflow(self, factvar):
        # Other tests often simply test output; here we test intermediate results.
        f = xfuncs.xfunc_covariance(factvar)

        null_matrix = [[NaN, NaN, NaN], [NaN, NaN, NaN], [NaN, NaN, NaN]]

        cube = xcube([])
        (covs,) = f.get_initial_regions(cube)
        assert arr_eq(covs, null_matrix)
        coordinates = None
        f.fill(coordinates, (covs,))
        assert arr_eq(covs, [[6.0, -12.5, NaN], [-12.5, 26.125, NaN], [NaN, NaN, NaN]])
        assert arr_eq(
            f.reduce(cube, (covs,)),
            [[6.0, -12.5, NaN], [-12.5, 26.125, NaN], [NaN, NaN, NaN]],
        )

        cube = xcube([arr1])
        (covs,) = f.get_initial_regions(cube)
        assert arr_eq(covs, [null_matrix, null_matrix])
        coordinates = reduce(operator.add, cube.strided_dims())
        f.fill(coordinates, (covs,))
        expected = [
            [
                [6.66666667, -13.33333333, 1.33333333],
                [-13.33333333, 26.66666667, -2.66666667],
                [1.33333333, -2.66666667, 1.66666667],
            ],
            [
                [6.66666667, -14.33333333, NaN],
                [-14.33333333, 30.91666667, NaN],
                [NaN, NaN, NaN],
            ],
        ]
        assert arr_eq(covs, expected)
        assert arr_eq(f.reduce(cube, (covs,)), expected,)

        cube = xcube([arr1, arr2])
        (covs,) = f.get_initial_regions(cube)
        assert arr_eq(covs, [[null_matrix, null_matrix], [null_matrix, null_matrix]])
        coordinates = reduce(operator.add, cube.strided_dims())
        f.fill(coordinates, (covs,))
        expected = [
            [
                [
                    [9.333333333333332, -18.666666666666664, 1.0],
                    [-18.666666666666664, 37.33333333333333, -2.0],
                    [1.0, -2.0, 1.0],
                ],
                null_matrix,
            ],
            [
                [[2.0, -4.0, -1.0], [-4.0, 8.0, 2.0], [-1.0, 2.0, 0.5]],
                [[18.0, -39.0, NaN], [-39.0, 84.5, NaN], [NaN, NaN, NaN]],
            ],
        ]
        assert arr_eq(covs, expected)
        assert arr_eq(f.reduce(cube, (covs,)), expected)

    def _test_weighted_workflow(self, factvar):
        # Other tests often simply test output; here we test intermediate results.
        f = xfuncs.xfunc_covariance(factvar, weights=wt)

        null_matrix = [[NaN, NaN, NaN], [NaN, NaN, NaN], [NaN, NaN, NaN]]

        cube = xcube([arr1, arr2])
        (covs,) = f.get_initial_regions(cube)
        assert arr_eq(covs, [[null_matrix, null_matrix], [null_matrix, null_matrix]])
        coordinates = reduce(operator.add, cube.strided_dims())
        f.fill(coordinates, (covs,))
        expected = [
            [
                [
                    [7.999999999999998, -15.999999999999996, -1.9999999999999996],
                    [-15.999999999999996, 31.999999999999993, 3.999999999999999],
                    [-1.9999999999999996, 3.999999999999999, 0.4999999999999999],
                ],
                null_matrix,
            ],
            [null_matrix, [[18.0, -39.0, NaN], [-39.0, 84.5, NaN], [NaN, NaN, NaN]],],
        ]

        assert arr_eq(covs, expected)
        assert arr_eq(f.reduce(cube, (covs,)), expected,)

    def test_single_arr_with_nan_workflow(self):
        factvar = [
            [-1, 1, 1],
            [-2, 4, 2],
            [-3, 6, 3],
            [-4, 8, 4],
            [-5, 10, 4],
            [-6, 12, 3],
            [-7, 14, NaN],
            [-8, 16, 1],
        ]
        self._test_unweighted_workflow(factvar)
        self._test_weighted_workflow(factvar)

    def test_tuple_factvar_workflow(self):
        factvar = (
            [
                [-1, 1, 1],
                [-2, 4, 2],
                [-3, 6, 3],
                [-4, 8, 4],
                [-5, 10, 4],
                [-6, 12, 3],
                [-7, 14, 2],
                [-8, 16, 1],
            ],
            [
                [True, True, True],
                [True, True, True],
                [True, True, True],
                [True, True, True],
                [True, True, True],
                [True, True, True],
                [True, True, False],
                [True, True, True],
            ],
        )
        self._test_unweighted_workflow(factvar)
        self._test_weighted_workflow(factvar)

    def test_tuple_factvar_with_nan_workflow(self):
        factvar = (
            [
                [-1, 1, 1],
                [-2, 4, 2],
                [-3, 6, 3],
                [-4, 8, 4],
                [-5, 10, 4],
                [-6, 12, 3],
                [-7, 14, NaN],
                [-8, 16, 1],
            ],
            [
                [True, True, True],
                [True, True, True],
                [True, True, True],
                [True, True, True],
                [True, True, True],
                [True, True, True],
                [True, True, False],
                [True, True, True],
            ],
        )
        self._test_unweighted_workflow(factvar)
        self._test_weighted_workflow(factvar)


class TestXfuncCovarianceIgnoreMissing:
    def test_ignore_missing(self):
        factvar = [
            [-1, 1, 1],
            [-2, 4, 2],
            [-3, 6, 3],
            [-4, 8, 4],
            [-5, 10, 4],
            [-6, 12, 3],
            [-7, 14, NaN],
            [-8, 16, 1],
        ]

        # The cube of arr1 has rowids:
        # 0           1
        # [1, 3, 5, 7]  [0, 2, 4, 6]

        covs = xcube([arr1]).covariance(factvar)
        assert arr_eq(
            covs,
            [
                [
                    [6.66666667, -13.33333333, 1.33333333],
                    [-13.33333333, 26.66666667, -2.66666667],
                    [1.33333333, -2.66666667, 1.66666667],
                ],
                # Cell (1,) MUST have missings, because it had a missing input.
                [
                    [6.66666667, -14.33333333, NaN],
                    [-14.33333333, 30.91666667, NaN],
                    [NaN, NaN, NaN],
                ],
            ],
        )
        covs = xcube([arr1]).covariance(factvar, ignore_missing=True)
        assert arr_eq(
            covs,
            [
                [
                    [6.66666667, -13.33333333, 1.33333333],
                    [-13.33333333, 26.66666667, -2.66666667],
                    [1.33333333, -2.66666667, 1.66666667],
                ],
                # Cell (1,) MUST NOT have missings, because we ignored any rows
                # with missings in the input, and it had at least one valid input.
                [
                    [4.0, -9.0, -3.0],
                    [-9.0, 20.33333333, 6.83333333],
                    [-3.0, 6.83333333, 2.33333333],
                ],
            ],
        )

        # The cube of arr1 x arr2 has fact values:
        #      ________arr 2________
        #      0                  1
        # a|0  [2.0, 6.0, 100.0]  [4.0]
        # r|
        # r|
        # 1|1  [nan, 5.0]     [1.0, 7.0]

        covs = xcube([arr1, arr2]).covariance(factvar)
        expected = [
            [
                [
                    [9.333333333333332, -18.666666666666664, 1.0],
                    [-18.666666666666664, 37.33333333333333, -2.0],
                    [1.0, -2.0, 1.0],
                ],
                [[NaN, NaN, NaN], [NaN, NaN, NaN], [NaN, NaN, NaN]],
            ],
            [
                [[2.0, -4.0, -1.0], [-4.0, 8.0, 2.0], [-1.0, 2.0, 0.5]],
                # Cell (1, 1) MUST have missings, because it had a missing input.
                [[18.0, -39.0, NaN], [-39.0, 84.5, NaN], [NaN, NaN, NaN]],
            ],
        ]
        assert arr_eq(covs, expected)

        covs = xcube([arr1, arr2]).covariance(factvar, ignore_missing=True)
        expected = [
            [
                [
                    [9.333333333333332, -18.666666666666664, 1.0],
                    [-18.666666666666664, 37.33333333333333, -2.0],
                    [1.0, -2.0, 1.0],
                ],
                [[NaN, NaN, NaN], [NaN, NaN, NaN], [NaN, NaN, NaN]],
            ],
            [
                [[2.0, -4.0, -1.0], [-4.0, 8.0, 2.0], [-1.0, 2.0, 0.5]],
                # Cell (1, 1) MUST NOT have missings, because we ignored any rows
                # with missings in the input, and it had at least one valid input.
                [[NaN, NaN, NaN], [NaN, NaN, NaN], [NaN, NaN, NaN]],
            ],
        ]


class TestXfuncCovarianceReturnMissingAs:
    def test_return_missing_as(self):
        factvar = [
            [-1, 1, 1],
            [-2, 4, 2],
            [-3, 6, 3],
            [-4, 8, 4],
            [-5, 10, 4],
            [-6, 12, 3],
            [-7, 14, NaN],
            [-8, 16, 1],
        ]

        covs = xcube([arr1]).covariance(factvar)
        assert arr_eq(
            covs,
            [
                [
                    [6.666666666666666, -13.333333333333332, 1.3333333333333333],
                    [-13.333333333333332, 26.666666666666664, -2.6666666666666665],
                    [1.3333333333333333, -2.6666666666666665, 1.6666666666666665],
                ],
                # Cell (1,) MUST have missings, because it had a missing input.
                [
                    [6.666666666666666, -14.333333333333332, NaN],
                    [-14.333333333333332, 30.916666666666664, NaN],
                    [NaN, NaN, NaN],
                ],
            ],
        )

        covs, validity = xcube([arr1]).covariance(factvar, return_missing_as=(0, False))
        assert arr_eq(
            covs,
            [
                [
                    [6.666666666666666, -13.333333333333332, 1.3333333333333333],
                    [-13.333333333333332, 26.666666666666664, -2.6666666666666665],
                    [1.3333333333333333, -2.6666666666666665, 1.6666666666666665],
                ],
                # Cell (1,) MUST have missings, because it had a missing input.
                [
                    [6.666666666666666, -14.333333333333332, 0.0],
                    [-14.333333333333332, 30.916666666666664, 0.0],
                    [0.0, 0.0, 0.0],
                ],
            ],
        )
        assert arr_eq(
            validity,
            [
                [[True, True, True], [True, True, True], [True, True, True]],
                # Cell (1,) MUST have missings, because it had a missing input.
                [[True, True, False], [True, True, False], [False, False, False]],
            ],
        )


class TestXfuncCovarianceEmptyCells:
    def test_empty_cells(self):
        # Use a dimension that has no entries for one of its values.
        dim1 = [2, 0, 2, 0, 2, 0, 2, 0]
        factvar = [
            [-1, 1, 1],
            [-2, 4, 2],
            [-3, 6, 3],
            [-4, 8, 4],
            [-5, 10, 4],
            [-6, 12, 3],
            [-7, 14, NaN],
            [-8, 16, 1],
        ]

        covs = xcube([dim1]).covariance(factvar)
        assert arr_eq(
            covs,
            [
                [
                    [6.666666666666666, -13.333333333333332, 1.3333333333333333],
                    [-13.333333333333332, 26.666666666666664, -2.6666666666666665],
                    [1.3333333333333333, -2.6666666666666665, 1.6666666666666665],
                ],
                [[NaN, NaN, NaN], [NaN, NaN, NaN], [NaN, NaN, NaN]],
                # Cell (2,) MUST have missings, because it had a missing input.
                [
                    [6.666666666666666, -14.333333333333332, NaN],
                    [-14.333333333333332, 30.916666666666664, NaN],
                    [NaN, NaN, NaN],
                ],
            ],
        )

        covs = xcube([dim1]).covariance(factvar, weights=wt)
        assert arr_eq(
            covs,
            [
                [
                    [2.947368421052631, -5.894736842105262, -0.42105263157894735],
                    [-5.894736842105262, 11.789473684210524, 0.8421052631578947],
                    [-0.42105263157894735, 0.8421052631578947, 0.9736842105263156],
                ],
                [[NaN, NaN, NaN], [NaN, NaN, NaN], [NaN, NaN, NaN]],
                [[NaN, NaN, NaN], [NaN, NaN, NaN], [NaN, NaN, NaN]],
            ],
        )
