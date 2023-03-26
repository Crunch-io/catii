import itertools
import multiprocessing.pool
import operator
from contextlib import closing
from functools import reduce

import numpy

from . import ffuncs
from .set_operations import set_intersect_merge_np

BIG_REGIONS = 1 << 30  # 1G, max input data size before we use threads <shrug>


class ccube:
    """An N-dimensional contingency cube of catii iindexes.

    This object defines a list of `dims`: iindex variables being crosstabbed.
    It also defines the .shape of the working and output arrays, and some
    slice objects to help navigate them.

    The ccube itself does not own the output; instead, use ffunc objects
    (or the shortcut methods on the ccube, like `count`) to calculate the
    aggregate data of a ccube. Multiple ffuncs may apply to a single ccube.

    The core operation on a ccube is interaction; that is, the grouping
    of rows by the combinations of distinct categorical dimensions. To work
    with higher dimensions, we iterate over the Cartesian product of them,
    take 1-D slices of each iindex, form a subcube for each, and stack those.
    To save time, we initialize "one big region" for each ffunc, representing
    the stacked output, then, after filling one subregion per subcube, do a
    single reduce operation on the stacked cube to perform all of the marginal
    differencing in one pass.
    """

    poolsize = 4
    debug = False
    check_interrupt = None
    _compare_output_to_xcube_for_tests = False

    def __init__(self, dims, interacting_shape=None):
        self.dims = dims

        self.scaffold_shape = tuple(e for d in dims for e in d.shape[1:])
        self.num_regions = reduce(operator.mul, self.scaffold_shape, 1)
        self.parallel = (
            self.num_regions > 2
            and (self.dims[0].shape[0] * self.num_regions) >= BIG_REGIONS
        )
        if interacting_shape is None:
            interacting_shape = tuple(
                max([coords[0] for coords in d] + [d.common]) + 1 for d in dims
            )
        self.interacting_shape = interacting_shape
        self.shape = self.scaffold_shape + self.interacting_shape
        self.working_shape = self.scaffold_shape + tuple(
            e + 1 for e in self.interacting_shape
        )

        self.scaffold = tuple(slice(None) for s in self.scaffold_shape)
        self.corner = self.scaffold + tuple(-1 for d in dims)
        self.marginless = self.scaffold + tuple(slice(0, -1) for dim in dims)

        self.intersection_data_points = 0

    # ----------------------------- interaction ----------------------------- #

    def _walk(self, dims, base_coords, base_rowids, funcs):
        # This method could be made smaller by moving the `if's` inside
        # the loops, but that actually becomes a performance issue when
        # you're looping over tens of thousands of subcubes.
        # Exploded like this can reduce completion time by 10%.
        if len(dims) > 1:
            remaining_dims = dims[1:]
            if base_rowids is None:
                # First dim, or the case where all higher dims have passed
                # ((-1,), None) to mean "ignore this dim".
                for coords, rowids in dims[0].items():
                    self._walk(remaining_dims, base_coords + coords, rowids, funcs)
            else:
                self.intersection_data_points += len(base_rowids) * len(dims[0])
                for coords, rowids in dims[0].items():
                    self.intersection_data_points += len(rowids)
                    rowids = set_intersect_merge_np(base_rowids, rowids)
                    if len(rowids):
                        self._walk(remaining_dims, base_coords + coords, rowids, funcs)

            # Margin
            self._walk(remaining_dims, base_coords + (-1,), base_rowids, funcs)
        elif dims:
            # Last dim. The `rowids` in each loop below are the rowids
            # that appear in all dims for this particular tuple of new_coords.
            # Any coordinate which is -1 targets the margin for that axis.
            if base_rowids is None:
                for coords, rowids in dims[0].items():
                    if len(rowids):
                        for func in funcs:
                            func(base_coords + coords, rowids)
            else:
                self.intersection_data_points += len(base_rowids) * len(dims[0])
                for coords, rowids in dims[0].items():
                    self.intersection_data_points += len(rowids)
                    rowids = set_intersect_merge_np(base_rowids, rowids)
                    if len(rowids):
                        for func in funcs:
                            func(base_coords + coords, rowids)

                # Margin
                if len(base_rowids):
                    for func in funcs:
                        func(base_coords + (-1,), base_rowids)

    def walk(self, func_or_funcs):
        """Call func(interaction-of-coords, intersection-of-rowids) over self.dims.

        This recursively iterates over each dim in self, combining coordinates
        from each along the way. For each distinct combination, we find the
        set of rowids which contain those coordinates--if there are any,
        we yield the combined coordinates and the rowids.

        COMMON VALUES ARE NOT INCLUDED. Instead, yielded coordinates
        with -1 values signify a marginal cell on that axis;
        the corresponding rowids are the set-intersection of all rowids
        in the other dimensions for the distinct coords without regard
        for their coordinate on those -1 axes. For example, if we yield:

            (3, -1, 2, -1): [4, 5, 9, 130]

        ...that means there are four rows in the input frame which have
        category 3 for dims[0] and category 2 for dims[2], regardless of
        what their categories are for dims 1 and 3. There are, therefore,
        four rows at (3, -1, 2, -1), where -1 is the margin along that axis.
        A simple count might then place a `4` in that cell in the cube;
        other ffuncs might use those rowids to look up other outputs.
        """
        if not isinstance(func_or_funcs, (tuple, list)):
            func_or_funcs = [func_or_funcs]
        self._walk(self.dims, (), None, func_or_funcs)

    def interactions(self):
        """Return (interaction-of-coords, intersection-of-rowids) pairs from self."""
        out = []
        self.walk([lambda c, r: out.append((c, r))])
        return out

    def _compute_common_cells_from_marginal_diffs(self, region):
        """Fill common cells by performing a "marginal diff" along each axis."""
        # That is, we subtract ccube.sum(axis=M) from each margin M.
        # Some of these will be computed multiple times if there are multiple
        # dimensions, but that's ok.
        #
        # For example, if we do a simple count of (X, Y) with the following data:
        #
        #   var X  var Y
        #       0      1 (respondent A)
        #       1      0 (respondent B)
        #
        # ...where 0 is the common value in each, then ccube.interactions will
        # form a 2x2 cube, plus an extra marginal row for the X axis (0)
        # and an extra marginal column for the Y axis (1):
        #
        #     [0,  0,    0],
        #     [0,  0,    1],
        #
        #     [0,  1,    2],
        #
        # It is a bit easier to show what's happening if we replace those
        # zeros with underscores, and the number 1's with the letter
        # of the respondent they represent:
        #
        #     [_,  _,    _],
        #     [_,  _,    b],
        #
        #     [_,  a,   ab],
        #
        # We expect the result to be (after cutting off the margins):
        #
        #     [_,  A],
        #     [B,  _],
        #
        # ...so, in effect, we need to "raise" the marginal "a" to its final
        # place "A" in the common row, and similarly for marginal "b"/common "B".
        #
        # If we start with axis 0, we "raise a to A" by a marginal diff:
        #
        #   `region[0] = region[-1, :] - region[:-1, :].sum(axis=0)`
        #
        # ...and we get:
        #
        #     [_,  A,   a],
        #     [_,  _,   b],
        #     [_,  a,  ab],
        #
        # Diffing that along axis 1, we get:
        #
        #     [_,  A,   a],
        #     [B,  _,   b],
        #     [b,  a,  ab],
        #
        # For data points whose coordinates include more than one common value,
        # we have to do a marginal diff of one axis, including the margin cells,
        # and then diff that along the next axis.
        axes = list(range(len(self.dims)))
        for axis in axes:
            common_slice = self.scaffold + tuple(
                dim.common if a == axis else slice(None)
                for a, dim in enumerate(self.dims)
            )
            margin_slice = self.scaffold + tuple(
                -1 if a == axis else slice(None) for a in axes
            )
            uncommon_slice = self.scaffold + tuple(
                slice(None, -1) if a == axis else slice(None) for a in axes
            )
            sumaxis = len(self.scaffold) + axis
            region[common_slice] = region[margin_slice] - region[uncommon_slice].sum(
                axis=sumaxis
            )

    # ------------------------------- stacking ------------------------------- #

    def product(self):
        """Cartesian product of coordinate dimensions.

        This returns an iterable: each element yielded from it is a set of
        1d "slices" from each dimension. For example, if our dimensions are
          * a 2-D iindex 'A' with 3 columns, then
          * a 1-D iindex 'B', then
          * a 3-D iindex 'C' with 2 columns and an additional axis of length 4,
        then this would generate:

            [{"coords": (0,), "data": A.sliced(0)}, {"coords": (), "data": B}, {"coords": (0, 0), "data": C.sliced(0, 0)}]
            [{"coords": (0,), "data": A.sliced(0)}, {"coords": (), "data": B}, {"coords": (0, 1), "data": C.sliced(0, 1)}]
            [{"coords": (0,), "data": A.sliced(0)}, {"coords": (), "data": B}, {"coords": (0, 2), "data": C.sliced(0, 2)}]
            [{"coords": (0,), "data": A.sliced(0)}, {"coords": (), "data": B}, {"coords": (0, 3), "data": C.sliced(0, 3)}]
            [{"coords": (0,), "data": A.sliced(0)}, {"coords": (), "data": B}, {"coords": (1, 0), "data": C.sliced(1, 0)}]
            [{"coords": (0,), "data": A.sliced(0)}, {"coords": (), "data": B}, {"coords": (1, 1), "data": C.sliced(1, 1)}]
            [{"coords": (0,), "data": A.sliced(0)}, {"coords": (), "data": B}, {"coords": (1, 2), "data": C.sliced(1, 2)}]
            ...
            [{"coords": (2,), "data": A.sliced(2)}, {"coords": (), "data": B}, {"coords": (1, 3), "data": C.sliced(1, 3)}]
            [{"coords": (2,), "data": A.sliced(2)}, {"coords": (), "data": B}, {"coords": (1, 4), "data": C.sliced(1, 4)}]

        We can then calculate aggregates for each subcube and fill them.
        If a dimension is already 1-D, like `B` in the above example, it is
        not sliced. Multiple multidimensional iindexes multiply the number
        of subcubes. Indexes with more than one higher dimension (like `C`
        in the above example) similarly multiply the number of subcubes.

        You may transpose the output afterward to match the desired order
        of dimensions.
        """
        # Wrap the (coords, slice) pairs in dicts so itertools.product
        # doesn't try to cross their internals.
        return itertools.product(
            *[
                ({"coords": c, "data": s} for c, s in dim.slices1d())
                for dim in self.dims
            ]
        )

    def calculate(self, funcs):
        """Return a tuple of aggregates, usually one NumPy array for each ffunc."""
        if self.debug:
            print("\nccube.calculate(%s):" % (funcs,))
            print(
                "scaffold_shape:",
                self.scaffold_shape,
                "interacting_shape:",
                self.interacting_shape,
            )
        results = [func.get_initial_regions(self) for func in funcs]
        if self.debug:
            print("INITIAL REGIONS:")
            for func, regions in zip(funcs, results):
                print(func, ":", regions)

        def fill_one_cube(subcube_dims):
            if self.check_interrupt is not None:
                self.check_interrupt()

            subcube = ccube(
                [dim["data"] for dim in subcube_dims],
                interacting_shape=self.interacting_shape,
            )
            subcube_coords = [dim["coords"] for dim in subcube_dims]
            if self.debug:
                print("FILL SUBCUBE:", subcube_coords)
            flattened_slice = [e for coords in subcube_coords for e in coords]

            fill_funcs = []
            for func, regions in zip(funcs, results):
                if flattened_slice:
                    # The coords, when concatenated together, define which region
                    # of the complete result array(s) should be filled in
                    # by aggregating the given 1d slices of input data.
                    # Form a view of this region to pass to each measure,
                    # so the ffuncs themselves don't have to know about
                    # our outer dimensions.
                    regions = [region[tuple(flattened_slice)] for region in regions]

                fill_funcs.append(func.fill_func(regions))
                if self.debug:
                    print(func, ":=", regions)

            subcube.walk(fill_funcs)
            self.intersection_data_points += subcube.intersection_data_points

        if self.parallel:
            with closing(multiprocessing.pool.ThreadPool(self.poolsize)) as pool:
                pool.map(fill_one_cube, self.product())
        else:
            # The only reason to _not_ multithread this is the extra overhead;
            # for example, if there's only one region anyway, or there are a handful
            # but we expect each to be very fast because the number of rows
            # is small.
            for subcube_dims in self.product():
                fill_one_cube(subcube_dims)

        output = [func.reduce(self, regions) for func, regions in zip(funcs, results)]
        if self.debug:
            print("OUTPUT:")
            for func, regions in zip(funcs, output):
                print(func, ":", regions)
        return output

    # -------------------------------- ffuncs -------------------------------- #

    def count(
        self, weights=None, N=None, ignore_missing=False, return_missing_as=ffuncs.NaN
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
        result = self.calculate(
            [ffuncs.ffunc_count(weights, N, ignore_missing, return_missing_as)]
        )[0]
        if self._compare_output_to_xcube_for_tests:
            from . import xcubes

            xcube = xcubes.xcube([d.to_array() for d in self.dims])
            xcube_result = xcube.count(weights, N, ignore_missing, return_missing_as)
            assert numpy.allclose(result, xcube_result, equal_nan=True)
        return result

    def valid_count(
        self, arr, weights=None, ignore_missing=False, return_missing_as=ffuncs.NaN
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
        result = self.calculate(
            [ffuncs.ffunc_valid_count(arr, weights, ignore_missing, return_missing_as)]
        )[0]
        if self._compare_output_to_xcube_for_tests:
            from . import xcubes

            xcube = xcubes.xcube([d.to_array() for d in self.dims])
            xcube_result = xcube.valid_count(
                arr, weights, ignore_missing, return_missing_as
            )
            assert numpy.allclose(result, xcube_result, equal_nan=True)
        return result

    def sum(
        self, arr, weights=None, ignore_missing=False, return_missing_as=ffuncs.NaN
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
        result = self.calculate(
            [ffuncs.ffunc_sum(arr, weights, ignore_missing, return_missing_as)]
        )[0]
        if self._compare_output_to_xcube_for_tests:
            from . import xcubes

            xcube = xcubes.xcube([d.to_array() for d in self.dims])
            xcube_result = xcube.sum(arr, weights, ignore_missing, return_missing_as)
            assert numpy.allclose(result, xcube_result, equal_nan=True)
        return result

    def mean(
        self, arr, weights=None, ignore_missing=False, return_missing_as=ffuncs.NaN
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
        result = self.calculate(
            [ffuncs.ffunc_mean(arr, weights, ignore_missing, return_missing_as)]
        )[0]
        if self._compare_output_to_xcube_for_tests:
            from . import xcubes

            xcube = xcubes.xcube([d.to_array() for d in self.dims])
            xcube_result = xcube.mean(arr, weights, ignore_missing, return_missing_as)
            assert numpy.allclose(result, xcube_result, equal_nan=True)
        return result
