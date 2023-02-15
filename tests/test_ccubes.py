import numpy

from catii import ccube, iindex
from catii.ffuncs import ffunc_count, ffunc_sum

from . import arr_eq


class TestCubeCreation:
    def test_direct_construction(self):
        idx1 = iindex({(1,): [0, 2, 7]}, 0, (8,))
        idx2 = iindex({(1,): [0, 2, 5]}, 0, (8,))
        cube = ccube([idx1, idx2])

        assert cube.dims == [idx1, idx2]
        assert cube.intersection_data_points == 0
        assert cube.shape == (2, 2)


class TestCubeDimensions:
    def test_cube_1d_x_1d(self):
        idx1 = iindex({(1,): [0, 2, 7]}, 0, (8,))
        idx2 = iindex({(1,): [0, 2, 5]}, 0, (8,))
        cube = ccube([idx1, idx2])
        assert cube.count().tolist() == [[4, 1], [1, 2]]


class TestCubeProduct:
    def test_cube_product(self):
        idx1 = iindex({(1,): [0, 2, 7]}, 0, (8,))
        cube = ccube([idx1, idx1])
        assert list(cube.product) == [(None, None)]

        idx2 = iindex({(1, 0): [0, 2, 5], (1, 1): [3, 4]}, 0, (8, 2))
        cube = ccube([idx1, idx2])
        assert list(cube.product) == [(None, (0,)), (None, (1,))]


class TestCubeSubcube:
    def test_cube_subcube(self):
        idx1 = iindex({(1,): [0, 2, 7]}, 0, (8,))
        cube = ccube([idx1, idx1])
        cube = cube.subcube([(), ()])
        assert cube.dims == [idx1, idx1]
        assert cube.intersection_data_points == 0
        assert cube.shape == (2, 2)

        idx2 = iindex({(1, 0): [0, 2, 5], (1, 1): [3, 4]}, 0, (8, 2))
        cube = ccube([idx1, idx2])
        assert cube.shape == (2, 2, 2)
        cube = cube.subcube([(), (1,)])
        assert cube.dims == [idx1, idx2.sliced(1)]
        assert cube.intersection_data_points == 0
        assert cube.shape == (2, 2)


class TestCubeCalculate:
    def test_cube_calculate(self):
        # [1, 0, 1, 0, 0, 0, 0, 1]
        idx1 = iindex({(1,): [0, 2, 7]}, 0, (8,))
        cube = ccube([idx1, idx1])
        counts = cube.calculate([ffunc_count()])[0]
        assert counts.tolist() == [[5, 0], [0, 3]]

        # 0: [1, 0, 1, 0, 0, 1, 0, 0],
        # 1: [0, 0, 0, 1, 1, 0, 0, 0]
        idx2 = iindex({(1, 0): [0, 2, 5], (1, 1): [3, 4]}, 0, (8, 2))
        cube = ccube([idx1, idx2])
        counts = cube.calculate([ffunc_count()])[0]
        assert counts.tolist() == [
            [[4, 1], [1, 2]],
            [[3, 2], [3, 0]],
        ]

        fsum = ffunc_sum((numpy.arange(8)))
        counts, sums = cube.calculate([ffunc_count(), fsum])
        assert counts.tolist() == [
            [[4, 1], [1, 2]],
            [[3, 2], [3, 0]],
        ]
        assert arr_eq(sums, [[[14, 5], [7, 2]], [[12, 7], [9, 0]],],)
