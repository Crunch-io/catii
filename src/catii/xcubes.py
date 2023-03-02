import itertools
import multiprocessing.pool
import operator
import time
from contextlib import closing
from functools import reduce

import numpy

from . import xfuncs

BIG_REGIONS = 1 << 30  # 1G, max input data size before we use threads <shrug>


class xcube:
    """An N-dimensional contingency cube of NumPy arrays.

    This object defines a list of `dims`: arrays being crosstabbed.
    It also defines the .shape of the working and output arrays, and some
    slice objects to help navigate them.

    The xcube itself does not own the output; instead, use xfunc objects
    (or the shortcut methods on the xcube, like `count`) to calculate the
    aggregate data of an xcube. Multiple xfuncs may apply to a single xcube.

    The core operation on an xcube is interaction; that is, the grouping
    of rows by the combinations of distinct categorical dimensions. To work
    with higher dimensions, we iterate over the Cartesian product of them,
    take 1-D slices of each dim, form a subcube for each, and stack those.
    To save time, we initialize "one big region" for each xfunc, representing
    the stacked output, then, after filling one subregion per subcube, do a
    single reduce operation on the stacked cube to perform all of the marginal
    differencing in one pass.
    """

    poolsize = 4
    debug = False
    check_interrupt = None
    pool_class = multiprocessing.pool.ThreadPool

    def __init__(self, dims, interacting_shape=None):
        self.dims = [numpy.asarray(d) for d in dims]

        self.scaffold_shape = tuple(e for d in self.dims for e in d.shape[1:])
        self.num_regions = reduce(operator.mul, self.scaffold_shape, 1)
        self.parallel = (
            self.num_regions > 2
            and (self.dims[0].shape[0] * self.num_regions) >= BIG_REGIONS
        )
        if interacting_shape is None:
            # Slow! Always pass interacting_shape if you already know extents.
            interacting_shape = tuple(max(d.flat) + 1 for d in self.dims)
        self.interacting_shape = interacting_shape
        self.shape = self.scaffold_shape + self.interacting_shape
        self._set_strides()

    def _set_strides(self):
        """Set self.multipliers/mintype.

        When calculating the interaction of multiple categorical dimensions,
        a shortcut is to multiply each one by the cumulative product of all
        the earlier ones, in effect pre-calculating the "stride" that dimension
        will take in the output arrays. For example, one dim of extent 2 and
        another of extent 3 will form a cube of shape (2, 3):

               dim 2
             __0___1__
        d  0 |_0_|_1_|
        i  1 |_2_|_3_|
        m  2 |_4_|_5_|
        1

        If we number the cells as if they were flattened/unraveled, they would
        be from 0-5, and we see that each cell number is (dim1 * 2) + dim2.
        We say dim1 has a "stride" or multiplier of 2. The method calculates
        self.multipliers. When we fill the cube, we multiply the category ids
        (contiguous integers starting from 0) by their stride.

        This also calculates the dtype needed to faithfully address the largest
        cell number as self.mintype.
        """
        multipliers = numpy.cumprod(list(reversed(self.interacting_shape)))
        self.multipliers = numpy.append(numpy.flip(multipliers)[1:], [1])

        # Coordinates might be of several dtypes: int8, int16 (even bool!).
        # If the cumulative product of shape exceeds the dtype
        # of any coords, we need to bump it up to at least that size
        # before doing additions and multiplications which might overflow.
        # In fact, if any coords.dtype is of a higher width than a
        # previous one, addition might fail with "Cannot cast" errors
        # from NumPy. Rather than supply casting rules to numpy.add
        # (which might hide real problems), we instead cast all coords
        # to the same type.
        maxmult = multipliers[-1] if len(multipliers) else 1
        for mintype in [numpy.uint8, numpy.uint16, numpy.uint32]:
            maxint = numpy.iinfo(mintype).max
            if maxmult <= maxint:
                break
        else:
            raise TypeError("Cannot calculate cubes with more than %s cells." % maxint)
        self.mintype = mintype

    # ----------------------------- interaction ----------------------------- #

    def strided_dims(self):
        """Return self.dims, multiplied by their stride for interacting."""
        return [
            dim.astype(self.mintype) if m == 1 else dim.astype(self.mintype) * m
            for m, dim in zip(self.multipliers, self.dims)
        ]

    # ------------------------------- stacking ------------------------------- #

    @property
    def product(self):
        """Cartesian product of coordinate dimensions.

        This returns an iterable of slice coordinates: each one a distinct
        combination describing 1-D slices from each dimension. For example,
        if our dimensions are a 2-D array 'A' with 3 columns, then a 1-D
        array 'B', then a 3-D array 'C' with 2 columns and an additional
        axis of length 4, then this would generate:

            (  # A     B     C
                ((0,), None, (0, 0)),  # -> (A.sliced(0) x B x C.sliced(0, 0))
                ((0,), None, (0, 1)),
                ((0,), None, (0, 2)),
                ((0,), None, (0, 3)),
                ((0,), None, (1, 0)),
                ((0,), None, (1, 1)),
                ((0,), None, (1, 2)),
                ...
                ((2,), None, (1, 3)),
                ((2,), None, (1, 4)),
            )

        We can then calculate aggregates for each xcube and fill them.
        If a dimension is already 1-D, like `B` in the above example,
        None is inserted. Multiple multidimensional dims multiply
        the number of xcubes. Arrays with more than one higher dimension
        (like `C` in the above example) similarly multiply the number of xcubes.

        You may transpose the output afterward to match the desired order
        of dimensions.
        """
        extents = []
        for d in self.dims:
            s = d.shape[1:]
            if s:
                extents.append(itertools.product(*[range(e) for e in s]))
            else:
                extents.append((None,))

        return itertools.product(*extents)

    def calculate(self, funcs):
        """Return a tuple of aggregates, usually one NumPy array for each xfunc."""
        if self.debug:
            print("\nxcube.calculate(%s):" % (funcs,))
        results = [func.get_initial_regions(self) for func in funcs]
        if self.debug:
            print("INITIAL REGIONS:")
            for func, regions in zip(funcs, results):
                print(func, ":", regions)

        strided_dims = self.strided_dims()

        self._tracing = {}
        for f in funcs:
            # Collect tracing for each ffunc (possibly running concurrently).
            self._tracing[f] = {"elapsed": 0.0, "start": None, "count": 0}

        def fill_one_cube(nested_coords):
            if self.check_interrupt is not None:
                self.check_interrupt()

            slices1d = [
                strided_dim if coords is None else strided_dim[(slice(None),) + coords]
                for coords, strided_dim in zip(nested_coords, strided_dims)
            ]

            # Obtain compound coordinates by simply adding across: the values have
            # already been multiplied by the proper stride.
            coordinates = reduce(operator.add, slices1d) if slices1d else None

            if self.debug:
                print("FILL SUBCUBE:", nested_coords)
            for func, regions in zip(funcs, results):
                start = time.time()

                flattened_slice = [
                    e for coords in nested_coords if coords is not None for e in coords
                ]
                if flattened_slice:
                    # The coords, when concatenated together, define which region
                    # of the complete result array(s) should be filled in
                    # by aggregating the given 1d slices of input data.
                    # Form a view of this region to pass to each measure,
                    # so the xfuncs themselves don't have to know about
                    # our outer dimensions.
                    regions = [region[tuple(flattened_slice)] for region in regions]

                func.fill(coordinates, regions)
                if self.debug:
                    print(func, ":=", regions)

                bucket = self._tracing[func]
                bucket["elapsed"] += time.time() - start
                bucket["count"] += 1
                if bucket["start"] is None:
                    bucket["start"] = start

        if self.parallel:
            with closing(self.pool_class(self.poolsize)) as pool:
                pool.map(fill_one_cube, self.product)
        else:
            # The only reason to _not_ multithread this is the extra overhead;
            # for example, if there's only one region anyway, or there are a handful
            # but we expect each to be very fast because the number of rows
            # is small.
            for nested_coords in self.product:
                fill_one_cube(nested_coords)

        output = [func.reduce(self, regions) for func, regions in zip(funcs, results)]
        if self.debug:
            print("OUTPUT:")
            for func, regions in zip(funcs, output):
                print(func, ":", regions)
        return output

    # -------------------------------- xfuncs -------------------------------- #

    def count(
        self, weights=None, N=None, ignore_missing=False, return_missing_as=xfuncs.NaN
    ):
        """Return the counts of self.dims.

        If `weights` is given and not None, it must be a NumPy array of numeric
        weight values, or a (weights, validity) tuple, corresponding row-wise
        to any cube.dims.

        If `ignore_missing` is False (the default), then any missing values
        are propagated so that outputs also have a missing value in any cell
        that had a missing value in the weights that contributed to that cell.
        If `ignore_missing` is True, such input rows are ignored and do not
        contribute to the output, much like NumPy's `nansum` or R's `na.rm = TRUE'.

        If `return_missing_as` is NaN (the default), the `reduce` method will
        return a single numeric NumPy array of counts. Any NaN values in it indicate
        missings: an output cell that had no inputs, or a NaN weight value,
        and therefore no count. If a 2-tuple like (0, False), the `reduce` method
        will return a NumPy array of counts, and a second "validity" NumPy array
        of booleans. Missing values will have 0 in the former and False in the
        latter.
        """
        return self.calculate(
            [xfuncs.xfunc_count(weights, N, ignore_missing, return_missing_as)]
        )[0]

    def valid_count(
        self, arr, weights=None, ignore_missing=False, return_missing_as=xfuncs.NaN
    ):
        """Return the valid counts of an array contingent on self.dims.

        The `arr` arg must be a NumPy array of numeric values to be counted,
        or a tuple of (values, validity) arrays, corresponding row-wise
        to any cube.dims.

        If `weights` is given and not None, it must be a NumPy array of numeric
        weight values, or a (weights, validity) tuple, corresponding row-wise
        to any cube.dims.

        If `ignore_missing` is False (the default), then any missing values
        are propagated so that outputs also have a missing value in any cell
        that had a missing value in one of the rows in the fact variable
        or weight that contributed to that cell. If `ignore_missing` is True,
        such input rows are ignored and do not contribute to the output,
        much like NumPy's `nansum` or R's `na.rm = TRUE'.

        If `return_missing_as` is NaN (the default), the `reduce` method will
        return a single numeric NumPy array of counts. Any NaN values in it
        indicate missing cells (an output cell that had no inputs, or a NaN
        weight value, and therefore no count). If `return_missing_as` is a 2-tuple,
        like (0, False), the `reduce` method will return a NumPy array of counts,
        and a second "validity" NumPy array of booleans. Missing values will have
        0 in the former and False in the latter.
        """
        return self.calculate(
            [xfuncs.xfunc_valid_count(arr, weights, ignore_missing, return_missing_as)]
        )[0]

    def sum(
        self, arr, weights=None, ignore_missing=False, return_missing_as=xfuncs.NaN
    ):
        """Return the sums of an array contingent on self.dims.

        The `arr` arg must be a NumPy array of numeric values to be summed,
        or a tuple of (values, validity) arrays, corresponding row-wise
        to any cube.dims.

        If `weights` is given and not None, it must be a NumPy array of numeric
        weight values, or a (weights, validity) tuple, corresponding row-wise
        to any cube.dims.

        If `ignore_missing` is False (the default), then any missing values
        are propagated so that outputs also have a missing value in any cell
        that had a missing value in one of the rows in the fact variable
        or weight that contributed to that cell. If `ignore_missing` is True,
        such input rows are ignored and do not contribute to the output,
        much like NumPy's `nansum` or R's `na.rm = TRUE'.

        If `return_missing_as` is NaN (the default), the `reduce` method will
        return a single numeric NumPy array of sums. Any NaN values in it
        indicate missing cells (an output cell that had no inputs, or a NaN
        weight value, and therefore no sum). If `return_missing_as` is a 2-tuple,
        like (0, False), the `reduce` method will return a NumPy array of sums,
        and a second "validity" NumPy array of booleans. Missing values will have
        0 in the former and False in the latter.
        """
        return self.calculate(
            [xfuncs.xfunc_sum(arr, weights, ignore_missing, return_missing_as)]
        )[0]

    def mean(
        self, arr, weights=None, ignore_missing=False, return_missing_as=xfuncs.NaN
    ):
        """Return the means of an array contingent on self.dims.

        The `arr` arg must be a NumPy array of numeric values to be meaned,
        or a tuple of (values, validity) arrays, corresponding row-wise
        to any cube.dims.

        If `weights` is given and not None, it must be a NumPy array of numeric
        weight values, or a (weights, validity) tuple, corresponding row-wise
        to any cube.dims.

        If `ignore_missing` is False (the default), then any missing values
        are propagated so that outputs also have a missing value in any cell
        that had a missing value in one of the rows in the fact variable
        or weight that contributed to that cell. If `ignore_missing` is True,
        such input rows are ignored and do not contribute to the output,
        much like NumPy's `nanmean` or R's `na.rm = TRUE'.

        If `return_missing_as` is NaN (the default), the `reduce` method will
        return a single numeric NumPy array of means. Any NaN values in it
        indicate missing cells (an output cell that had no inputs, or a NaN
        weight value, and therefore no mean). If `return_missing_as` is a 2-tuple,
        like (0, False), the `reduce` method will return a NumPy array of means,
        and a second "validity" NumPy array of booleans. Missing values will have
        0 in the former and False in the latter.
        """
        return self.calculate(
            [xfuncs.xfunc_mean(arr, weights, ignore_missing, return_missing_as)]
        )[0]

    def stddev(
        self, arr, weights=None, ignore_missing=False, return_missing_as=xfuncs.NaN
    ):
        """Return the standard deviations of an array contingent on self.dims.

        The `arr` arg must be a NumPy array of numeric values to be stddev'ed,
        or a tuple of (values, validity) arrays, corresponding row-wise
        to any cube.dims.

        If `weights` is given and not None, it must be a NumPy array of numeric
        weight values, or a (weights, validity) tuple, corresponding row-wise
        to any cube.dims.

        If `ignore_missing` is False (the default), then any missing values
        are propagated so that outputs also have a missing value in any cell
        that had a missing value in one of the rows in the fact variable
        or weight that contributed to that cell. If `ignore_missing` is True,
        such input rows are ignored and do not contribute to the output,
        much like NumPy's `nanstd` or R's `na.rm = TRUE'.

        If `return_missing_as` is NaN (the default), the `reduce` method will
        return a single numeric NumPy array of stddevs. Any NaN values in it
        indicate missing cells (an output cell that had no inputs, or a NaN
        weight value, and therefore no stddev). If `return_missing_as` is a 2-tuple,
        like (0, False), the `reduce` method will return a NumPy array of stddevs,
        and a second "validity" NumPy array of booleans. Missing values will have
        0 in the former and False in the latter.
        """
        return self.calculate(
            [xfuncs.xfunc_stddev(arr, weights, ignore_missing, return_missing_as)]
        )[0]

    def quantile(
        self,
        arr,
        probability,
        weights=None,
        ignore_missing=False,
        return_missing_as=xfuncs.NaN,
    ):
        """Return the q-quantile of an array contingent on self.dims.

        The `arr` arg must be a NumPy array of numeric values to be analyzed,
        or a tuple of (values, validity) arrays, corresponding row-wise
        to any cube.dims.

        The `probability` arg must be a number between 0 and 1 inclusive. To obtain
        the k'th q-quantile, pass `probability = k/q`. For example, to find the
        median, pass 0.5; to find the last quintile, pass 4/5 = 0.8.

        If `weights` is given and not None, it must be a NumPy array of numeric
        weight values, or a (weights, validity) tuple, corresponding row-wise
        to any cube.dims.

        If `ignore_missing` is False (the default), then any missing values
        are propagated so that outputs also have a missing value in any cell
        that had a missing value in one of the rows in the fact variable
        or weight that contributed to that cell. If `ignore_missing` is True,
        such input rows are ignored and do not contribute to the output,
        much like NumPy's `nan*` functions or R's `na.rm'.

        If `return_missing_as` is NaN (the default), the `reduce` method will
        return a single numeric NumPy array of quantiles, one per distinct
        combination of the cube.dims. Any NaN values in it indicate missing cells
        (an output cell that had no inputs, or a NaN weight value, and therefore
        no output). If `return_missing_as` is a 2-tuple, like (0, False), the
        `reduce` method will return a NumPy array of quantiles, and a second
        "validity" NumPy array of booleans. Missing values will have 0 in the
        former and False in the latter.
        """
        return self.calculate(
            [
                xfuncs.xfunc_quantile(
                    arr, probability, weights, ignore_missing, return_missing_as
                )
            ]
        )[0]

    def max(self, arr, ignore_missing=False, return_missing_as=xfuncs.NaN):
        """Return the maximums of an array contingent on self.dims.

        The `arr` arg must be a NumPy array of values, or a tuple of
        (values, validity) arrays, corresponding row-wise to any cube.dims.

        If `ignore_missing` is False (the default), then any missing values
        are propagated so that outputs also have a missing value in any cell
        that had a missing value in one of the rows in the fact variable
        or weight that contributed to that cell. If `ignore_missing` is True,
        such input rows are ignored and do not contribute to the output,
        much like NumPy's `nanmean` or R's `na.rm = TRUE'.

        If `return_missing_as` is NaN (the default), the `reduce` method will
        return a single numeric NumPy array of maximums. Any NaN values in it
        indicate missing cells (an output cell that had no inputs, and
        therefore no max). If `return_missing_as` is a 2-tuple, like (0, False),
        the `reduce` method will return a NumPy array of maximums, and a second
        "validity" NumPy array of booleans. Missing values will have 0 in the
        former and False in the latter.
        """
        return self.calculate(
            [xfuncs.xfunc_max(arr, ignore_missing, return_missing_as)]
        )[0]

    def min(self, arr, ignore_missing=False, return_missing_as=xfuncs.NaN):
        """Return the minimums of an array contingent on self.dims.

        The `arr` arg must be a NumPy array of values, or a tuple of
        (values, validity) arrays, corresponding row-wise to any cube.dims.

        If `ignore_missing` is False (the default), then any missing values
        are propagated so that outputs also have a missing value in any cell
        that had a missing value in one of the rows in the fact variable
        or weight that contributed to that cell. If `ignore_missing` is True,
        such input rows are ignored and do not contribute to the output,
        much like NumPy's `nanmean` or R's `na.rm = TRUE'.

        If `return_missing_as` is NaN (the default), the `reduce` method will
        return a single numeric NumPy array of minimums. Any NaN values in it
        indicate missing cells (an output cell that had no inputs, and
        therefore no min). If `return_missing_as` is a 2-tuple, like (0, False),
        the `reduce` method will return a NumPy array of minimums, and a second
        "validity" NumPy array of booleans. Missing values will have 0 in the
        former and False in the latter.
        """
        return self.calculate(
            [xfuncs.xfunc_min(arr, ignore_missing, return_missing_as)]
        )[0]

    def corrcoef(
        self, arr, weights=None, ignore_missing=False, return_missing_as=xfuncs.NaN
    ):
        """Return the correlation coefficients of an array contingent on self.dims.

        The `arr` arg must be a NumPy array of numeric values to be analyzed,
        or a tuple of (values, validity) arrays, corresponding row-wise
        to any cube.dims.

        If `weights` is given and not None, it must be a NumPy array of numeric
        weight values, or a (weights, validity) tuple, corresponding row-wise
        to any cube.dims.

        If `ignore_missing` is False (the default), then any missing values
        are propagated so that outputs also have a missing value in any cell
        that had a missing value in one of the rows in the fact variable
        or weight that contributed to that cell. If `ignore_missing` is True,
        such input rows are ignored and do not contribute to the output,
        much like NumPy's `nan*` functions or R's `use=na.or.complete'.

        If `return_missing_as` is NaN (the default), the `reduce` method will
        return a single numeric NumPy array of coefficient matrixes, one matrix
        of shape C * C, where C is the number of columns in the array, per distinct
        combination of the cube.dims. Any NaN values in it indicate missing cells
        (an output cell that had no inputs, or a NaN weight value, and therefore
        no output). If `return_missing_as` is a 2-tuple, like (0, False), the
        `reduce` method will return a NumPy array of matrixes, and a second
        "validity" NumPy array of booleans. Missing values will have 0 in the
        former and False in the latter.
        """
        return self.calculate(
            [xfuncs.xfunc_corrcoef(arr, weights, ignore_missing, return_missing_as)]
        )[0]

    def covariance(
        self, arr, weights=None, ignore_missing=False, return_missing_as=xfuncs.NaN
    ):
        """Return the covariance matrixes of an array contingent on self.dims.

        The `arr` arg must be a NumPy array of numeric values to be analyzed,
        or a tuple of (values, validity) arrays, corresponding row-wise
        to any cube.dims.

        If `weights` is given and not None, it must be a NumPy array of numeric
        weight values, or a (weights, validity) tuple, corresponding row-wise
        to any cube.dims.

        If `ignore_missing` is False (the default), then any missing values
        are propagated so that outputs also have a missing value in any cell
        that had a missing value in one of the rows in the fact variable
        or weight that contributed to that cell. If `ignore_missing` is True,
        such input rows are ignored and do not contribute to the output,
        much like NumPy's `nan*` functions or R's `use=na.or.complete'.

        If `return_missing_as` is NaN (the default), the `reduce` method will
        return a single numeric NumPy array of covariance matrixes, one matrix
        of shape C * C, where C is the number of columns in the array, per distinct
        combination of the cube.dims. Any NaN values in it indicate missing cells
        (an output cell that had no inputs, or a NaN weight value, and therefore
        no output). If `return_missing_as` is a 2-tuple, like (0, False), the
        `reduce` method will return a NumPy array of matrixes, and a second
        "validity" NumPy array of booleans. Missing values will have 0 in the
        former and False in the latter.
        """
        return self.calculate(
            [xfuncs.xfunc_covariance(arr, weights, ignore_missing, return_missing_as)]
        )[0]
