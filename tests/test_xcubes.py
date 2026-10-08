import numpy

from catii import xcube
from catii.xfuncs import xfunc_count, xfunc_mean, xfunc_stddev, xfunc_sum

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
        assert arr_eq(counts, [[5, float("nan")], [float("nan"), 3]])

        # 0: [1, 0, 1, 0, 0, 1, 0, 0],
        # 1: [0, 0, 0, 1, 1, 0, 0, 0]
        arr2 = [[1, 0], [0, 0], [1, 0], [0, 1], [0, 1], [1, 0], [0, 0], [0, 0]]
        cube = xcube([arr1, arr2])
        counts = cube.calculate([xfunc_count()])[0]
        assert arr_eq(counts, [[[4, 1], [1, 2]], [[3, 2], [3, float("nan")]]])

        fsum = xfunc_sum(numpy.arange(8))
        counts, sums = cube.calculate([xfunc_count(), fsum])
        assert arr_eq(counts, [[[4, 1], [1, 2]], [[3, 2], [3, float("nan")]]])
        assert arr_eq(sums, [[[14, 5], [7, 2]], [[12, 7], [9, float("nan")]]])

    def test_xcube_calculate_mean_and_stddev(self):
        """Test calculating mean and stddev together in one cube pass.

        Both xfunc_mean and xfunc_stddev compute weighted sums, weighted counts,
        and means internally. This test verifies they can be calculated together,
        serving as a baseline for potential optimization to share these computations.
        """
        arr1 = [1, 0, 1, 0, 0, 0, 0, 1]  # dim with values 0 and 1
        # 2D second dimension:
        # 0: [1, 0, 1, 0, 0, 1, 0, 0],
        # 1: [0, 0, 0, 1, 1, 0, 0, 0]
        arr2 = [[1, 0], [0, 0], [1, 0], [0, 1], [0, 1], [1, 0], [0, 0], [0, 0]]

        # Fact variable: values 0-7
        factvar = numpy.arange(8, dtype=float)

        cube = xcube([arr1, arr2])

        # Calculate mean and stddev together
        fmean = xfunc_mean(factvar)
        fstddev = xfunc_stddev(factvar)
        means, stddevs = cube.calculate([fmean, fstddev])

        # The cube of arr1 x arr2[col] has these rowids:
        # arr1=[1,0,1,0,0,0,0,1] and arr2 col 0=[1,0,1,0,0,1,0,0]
        # Row 0: arr1=1, arr2[:,0]=1 -> (1,1)
        # Row 1: arr1=0, arr2[:,0]=0 -> (0,0)
        # Row 2: arr1=1, arr2[:,0]=1 -> (1,1)
        # Row 3: arr1=0, arr2[:,0]=0 -> (0,0)
        # Row 4: arr1=0, arr2[:,0]=0 -> (0,0)
        # Row 5: arr1=0, arr2[:,0]=1 -> (0,1)
        # Row 6: arr1=0, arr2[:,0]=0 -> (0,0)
        # Row 7: arr1=1, arr2[:,0]=0 -> (1,0)
        #
        # For col 0:
        #        arr2[:,0]=0     arr2[:,0]=1
        # arr1=0  [1,3,4,6]       [5]
        # arr1=1  [7]             [0,2]
        #
        # For col 1 (arr2 col 1=[0,0,0,1,1,0,0,0]):
        #        arr2[:,1]=0     arr2[:,1]=1
        # arr1=0  [1,5,6]         [3,4]
        # arr1=1  [0,2,7]         []  (empty)

        # Output shape is [col, arr1, arr2_value] = [2, 2, 2]
        # Per the existing test, higher axes of 2D dims become outermost.

        # Col 0:
        # (arr1=0, arr2[:,0]=0): rows [1,3,4,6], factvar=[1,3,4,6], mean=3.5
        # (arr1=0, arr2[:,0]=1): rows [5], factvar=[5], mean=5.0
        # (arr1=1, arr2[:,0]=0): rows [7], factvar=[7], mean=7.0
        # (arr1=1, arr2[:,0]=1): rows [0,2], factvar=[0,2], mean=1.0

        # Col 1:
        # (arr1=0, arr2[:,1]=0): rows [1,5,6], factvar=[1,5,6], mean=4.0
        # (arr1=0, arr2[:,1]=1): rows [3,4], factvar=[3,4], mean=3.5
        # (arr1=1, arr2[:,1]=0): rows [0,2,7], factvar=[0,2,7], mean=3.0
        # (arr1=1, arr2[:,1]=1): empty -> NaN

        expected_means = [
            # col 0
            [[3.5, 5.0], [7.0, 1.0]],
            # col 1
            [[4.0, 3.5], [3.0, float("nan")]],
        ]
        assert arr_eq(means, expected_means)

        # Stddev requires at least 2 values, so single-element cells are NaN.
        # Col 0:
        # (arr1=0, arr2[:,0]=0): std([1,3,4,6], ddof=1)
        # (arr1=0, arr2[:,0]=1): std([5]) -> NaN (n=1)
        # (arr1=1, arr2[:,0]=0): std([7]) -> NaN (n=1)
        # (arr1=1, arr2[:,0]=1): std([0,2], ddof=1)

        # Col 1:
        # (arr1=0, arr2[:,1]=0): std([1,5,6], ddof=1)
        # (arr1=0, arr2[:,1]=1): std([3,4], ddof=1)
        # (arr1=1, arr2[:,1]=0): std([0,2,7], ddof=1)
        # (arr1=1, arr2[:,1]=1): empty -> NaN

        expected_stddevs = [
            # col 0
            [
                [numpy.std([1, 3, 4, 6], ddof=1), float("nan")],
                [float("nan"), numpy.std([0, 2], ddof=1)],
            ],
            # col 1
            [
                [numpy.std([1, 5, 6], ddof=1), numpy.std([3, 4], ddof=1)],
                [numpy.std([0, 2, 7], ddof=1), float("nan")],
            ],
        ]
        assert arr_eq(stddevs, expected_stddevs)
