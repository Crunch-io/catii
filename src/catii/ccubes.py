import itertools
import multiprocessing.pool
import operator
import time
from contextlib import closing
from functools import reduce

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

    def __init__(self, dims, interacting_shape=None):
        self.dims = dims

        self.scaffold_shape = tuple(e for d in dims for e in d.shape[1:])
        self.num_regions = reduce(operator.mul, self.scaffold_shape, 1)
        self.parallel = (
            self.num_regions > 2
            and (self.dims[0].shape[0] * self.num_regions) >= BIG_REGIONS
        )
        if interacting_shape is None:
            interacting_shape = tuple(max(coords[0] for coords in d) + 1 for d in dims)
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

    def _walk(self, dims, base_coords, base_set, func):
        # This method could be made smaller by moving the `if's` inside
        # the loops, but that actually becomes a performance issue when
        # you're looping over tens of thousands of subvariables.
        # Exploded like this can reduce completion time by 10%.
        remaining_dims = dims[1:]
        if remaining_dims:
            if base_set is None:
                # First dim, or the case where all higher dims have passed
                # ((-1,), None) to mean "ignore this dim".
                for coords, rowids in dims[0].items():
                    self._walk(remaining_dims, base_coords + coords, rowids, func)
            else:
                for coords, rowids in dims[0].items():
                    self.intersection_data_points += len(base_set) + len(rowids)
                    rowids = set_intersect_merge_np(base_set, rowids)
                    if rowids.shape[0]:
                        self._walk(remaining_dims, base_coords + coords, rowids, func)

            # Margin
            self._walk(remaining_dims, base_coords + (-1,), base_set, func)
        elif dims:
            # Last dim. The `rowids` in each loop below are the rowids
            # that appear in all dims for this particular tuple of new_coords.
            # Any coordinate which is -1 targets the margin for that axis.
            if base_set is None:
                for coords, rowids in dims[0].items():
                    if rowids.shape[0]:
                        func(base_coords + coords, rowids)
            else:
                for coords, rowids in dims[0].items():
                    self.intersection_data_points += len(base_set) + len(rowids)
                    rowids = set_intersect_merge_np(base_set, rowids)
                    if rowids.shape[0]:
                        func(base_coords + coords, rowids)

                # Margin
                if base_set.shape[0]:
                    func(base_coords + (-1,), base_set)

    def walk(self, func):
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
        self._walk(self.dims, (), None, func)

    def interactions(self):
        """Return (interaction-of-coords, intersection-of-rowids) pairs from self."""
        out = []
        self.walk(lambda c, r: out.append((c, r)))
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

    @property
    def product(self):
        """Cartesian product of coordinate dimensions.

        This returns an iterable of slice coordinates: each one a distinct
        combination describing 1-D slices from each dimension. For example,
        if our dimensions are a 2-D iindex 'A' with 3 columns, then a 1-D
        iindex 'B', then a 3-D iindex 'C' with 2 columns and an additional
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

        We can then calculate aggregates for each ccube and fill them.
        If a dimension is already 1-D, like `B` in the above example,
        None is inserted. Multiple multidimensional iindexes multiply
        the number of ccubes. Indexes with more than one higher dimension
        (like `C` in the above example) similarly multiply the number of ccubes.

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

    def subcube(self, nested_coords):
        return ccube(
            [
                dim if coords is None else dim.sliced(*coords)
                for coords, dim in zip(nested_coords, self.dims)
            ],
            interacting_shape=self.interacting_shape,
        )

    def calculate(self, funcs):
        """Return a tuple of aggregates, usually one NumPy array for each ffunc."""
        if self.debug:
            print("\nccube.calculate(%s):" % (funcs,))
        results = [func.get_initial_regions(self) for func in funcs]
        if self.debug:
            print("INITIAL REGIONS:")
            for func, regions in zip(funcs, results):
                print(func, ":", regions)

        self._tracing = {}
        for f in funcs:
            # Collect tracing for each ffunc (possibly running concurrently).
            self._tracing[f] = {"elapsed": 0.0, "start": None, "count": 0}

        def fill_one_cube(nested_coords):
            if self.check_interrupt is not None:
                self.check_interrupt()

            subcube = self.subcube(nested_coords)
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
                    # so the ffuncs themselves don't have to know about
                    # our outer dimensions.
                    regions = [region[tuple(flattened_slice)] for region in regions]

                func.fill(subcube, regions)
                self.intersection_data_points += subcube.intersection_data_points
                if self.debug:
                    print(func, ":=", regions)

                bucket = self._tracing[func]
                bucket["elapsed"] += time.time() - start
                bucket["count"] += 1
                if bucket["start"] is None:
                    bucket["start"] = start

        if self.parallel:
            with closing(multiprocessing.pool.ThreadPool(self.poolsize)) as pool:
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
        return self.calculate(
            [ffuncs.ffunc_count(weights, N, ignore_missing, return_missing_as)]
        )[0]

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
        return self.calculate(
            [ffuncs.ffunc_valid_count(arr, weights, ignore_missing, return_missing_as)]
        )[0]

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
        return self.calculate(
            [ffuncs.ffunc_sum(arr, weights, ignore_missing, return_missing_as)]
        )[0]

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
        return self.calculate(
            [ffuncs.ffunc_mean(arr, weights, ignore_missing, return_missing_as)]
        )[0]
