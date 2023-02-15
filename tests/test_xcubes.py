import numpy

from catii import xcube
from catii.xfuncs import xfunc_count, xfunc_sum

from . import arr_eq


class TestXCubeCreation:
    def test_direct_construction(self):
        arr1 = [1, 0, 1, 0, 0, 0, 0, 1]
        arr2 = [1, 0, 1, 0, 0, 1, 0, 0]
        cube = xcube([arr1, arr2])

        # assert cube.dims == [arr1, arr2]
        assert cube.shape == (2, 2)


class TestXCubeDimensions:
    def test_xcube_1d_x_1d(self):
        arr1 = [1, 0, 1, 0, 0, 0, 0, 1]
        arr2 = [1, 0, 1, 0, 0, 1, 0, 0]
        cube = xcube([arr1, arr2])
        assert cube.count().tolist() == [[4, 1], [1, 2]]


class TestXCubeProduct:
    def test_xcube_product(self):
        arr1 = [1, 0, 1, 0, 0, 0, 0, 1]
        cube = xcube([arr1, arr1])
        assert list(cube.product) == [(None, None)]

        arr2 = [[1, 0], [0, 0], [1, 0], [0, 1], [0, 1], [1, 0], [0, 0], [0, 0], [0, 0]]
        cube = xcube([arr1, arr2])
        assert list(cube.product) == [(None, (0,)), (None, (1,))]


class TestXCubeStridedDims:
    def test_xcube_strided_dims(self):
        arr1 = numpy.array([0, 3, 100], dtype=numpy.int8)
        cube = xcube([arr1, arr1])
        dim1, dim2 = cube.strided_dims()
        assert dim1.tolist() == [0, 3 * 101, 100 * 101]
        assert dim2.tolist() == [0, 3, 100]
        assert dim1.dtype == numpy.uint16().dtype
        assert dim2.dtype == numpy.uint16().dtype


class TestXCubeCalculate:
    def test_xcube_calculate(self):
        # [1, 0, 1, 0, 0, 0, 0, 1]
        arr1 = [1, 0, 1, 0, 0, 0, 0, 1]
        cube = xcube([arr1, arr1])
        counts = cube.calculate([xfunc_count()])[0]
        assert counts.tolist() == [[5, 0], [0, 3]]

        # 0: [1, 0, 1, 0, 0, 1, 0, 0],
        # 1: [0, 0, 0, 1, 1, 0, 0, 0]
        arr2 = [[1, 0], [0, 0], [1, 0], [0, 1], [0, 1], [1, 0], [0, 0], [0, 0]]
        cube = xcube([arr1, arr2])
        counts = cube.calculate([xfunc_count()])[0]
        assert counts.tolist() == [
            [[4, 1], [1, 2]],
            [[3, 2], [3, 0]],
        ]

        fsum = xfunc_sum(numpy.arange(8))
        counts, sums = cube.calculate([xfunc_count(), fsum])
        assert counts.tolist() == [
            [[4, 1], [1, 2]],
            [[3, 2], [3, 0]],
        ]
        assert arr_eq(sums, [[[14, 5], [7, 2]], [[12, 7], [9, 0]],],)
