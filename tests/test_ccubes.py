import numpy

from catii import ccube, iindex
from catii.ffuncs import ffunc_count, ffunc_sum

from . import arr_eq, compare_ccube_to_xcube


class TestCubeCreation:
    def test_direct_construction(self):
        idx1 = iindex({(1,): [0, 2, 7]}, 0, (8,))
        idx2 = iindex({(1,): [0, 2, 5]}, 0, (8,))
        cube = ccube([idx1, idx2])

        assert cube.dims == [idx1, idx2]
        assert cube.intersection_data_points == 0
        assert cube.shape == (2, 2)

    def test_explicit_shape_arg(self):
        idx1 = iindex({(1,): [0, 2, 7]}, 0, (8,))
        idx2 = iindex({(1,): [0, 2, 5]}, 0, (8,))

        cube = ccube([idx1, idx2], interacting_shape=(2, 2))
        assert cube.dims == [idx1, idx2]
        assert cube.shape == (2, 2)

        cube = ccube([idx1, idx2], interacting_shape=(4, 3))
        assert cube.dims == [idx1, idx2]
        assert cube.shape == (4, 3)

    def test_implicit_shape_arg(self):
        # Construct indexes where the common value is the highest value,
        # to make sure we are not only looking at index values to infer shape.
        idx1 = iindex({(0,): [0, 2, 7]}, 2, (8,))
        idx2 = iindex({(0,): [0, 2, 5]}, 3, (8,))
        cube = ccube([idx1, idx2])

        assert cube.dims == [idx1, idx2]
        assert cube.shape == (3, 4)


class TestCubeDimensions:
    @compare_ccube_to_xcube
    def test_cube_1d_x_1d(self):
        idx1 = iindex({(1,): [0, 2, 7]}, 0, (8,))
        idx2 = iindex({(1,): [0, 2, 5]}, 0, (8,))
        cube = ccube([idx1, idx2])
        assert cube.count().tolist() == [[4, 1], [1, 2]]


class TestCubeProduct:
    def test_cube_product(self):
        idx1 = iindex({(1,): [0, 2, 7]}, 0, (8,))
        cube = ccube([idx1, idx1])
        result = list(cube.product())
        for subcube in result:
            for dim in subcube:
                dim["data"] = {k: v.tolist() for k, v in dim["data"].items()}
        assert result == [
            (
                {"coords": (), "data": {(1,): [0, 2, 7]}},
                {"coords": (), "data": {(1,): [0, 2, 7]}},
            )
        ]

        idx2 = iindex({(1, 0): [0, 2, 5], (1, 1): [3, 4]}, 0, (8, 2))
        cube = ccube([idx1, idx2])
        result = list(cube.product())
        for subcube in result:
            for dim in subcube:
                dim["data"] = {
                    k: v if isinstance(v, list) else v.tolist()
                    for k, v in dim["data"].items()
                }
        assert result == [
            (
                {"coords": (), "data": {(1,): [0, 2, 7]}},
                {"coords": (0,), "data": {(1,): [0, 2, 5]}},
            ),
            (
                {"coords": (), "data": {(1,): [0, 2, 7]}},
                {"coords": (1,), "data": {(1,): [3, 4]}},
            ),
        ]


class TestCubeCalculate:
    def test_cube_calculate(self):
        # [1, 0, 1, 0, 0, 0, 0, 1]
        idx1 = iindex({(1,): [0, 2, 7]}, 0, (8,))
        cube = ccube([idx1, idx1])
        counts = cube.calculate([ffunc_count()])[0]
        assert arr_eq(counts, [[5, float("nan")], [float("nan"), 3]])

        # 0: [1, 0, 1, 0, 0, 1, 0, 0],
        # 1: [0, 0, 0, 1, 1, 0, 0, 0]
        idx2 = iindex({(1, 0): [0, 2, 5], (1, 1): [3, 4]}, 0, (8, 2))
        cube = ccube([idx1, idx2])
        counts = cube.calculate([ffunc_count()])[0]
        assert arr_eq(counts, [[[4, 1], [1, 2]], [[3, 2], [3, float("nan")]],])

        fsum = ffunc_sum((numpy.arange(8)))
        counts, sums = cube.calculate([ffunc_count(), fsum])
        assert arr_eq(counts, [[[4, 1], [1, 2]], [[3, 2], [3, float("nan")]],])
        assert arr_eq(sums, [[[14, 5], [7, 2]], [[12, 7], [9, 0]]])
