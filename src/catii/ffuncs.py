"""Frequency (and other aggregate) functions for catii.

These take cubes as input and return NumPy arrays as output from their `reduce`
methods, In between, they create, fill, and reduce "regions" (NumPy arrays)
as intermediate workspaces. The number of regions and their dtype depends on the
function: ffunc_count, for example, uses one region of ints or floats and mostly
returns it unchanged, while ffunc_mean uses separate "sums" and "valid_counts"
regions as the numerator and denominator, dividing them in its "reduce" method
to return a single array of means.

Most arguments to ffuncs (including weights) are variables which correspond
row-wise with cube dims, and must therefore have the same number of rows.
All such arguments may take one of two forms:
    * a single NumPy array, where missing values are represented by NaN or NaT
    * a pair of NumPy arrays, where the first contains values and the second
      is a "validity" array of booleans: True meaning "valid" and False meaning
      "missing". Where False, the corresponding values in the first array
      are ignored.

To some extent, which format you choose depends on your application and how
your data is already represented. Note, however, that NumPy arrays of `int`
or `str` have no standard way to represent missing values. Rather than nominate
sentinel values for these and similar types, you may pass a separate "validity"
array of booleans, and you might therefore consider doing so for all dtypes.
Note this is slightly faster, as well.

Functions defined here all have a `reduce` method which returns cube output
as NumPy arrays; these outputs may also have a missing value in any cell that
1) had no rows in the inputs with cube dimensions corresponding to it, or
2) had rows but corresponding fact values or weights were missing values ("all"
if `ignore_missing` else "any").

This is somewhat divergent from standard NumPy which, for example, defaults
numpy.sum([]) to 0.0. However, consumers of catii often need to distinguish
a sum or mean of [0, 0] from one of []. You are, of course, free to take the
output and set arr[arr.isnan()] = 0 if you desire.

Each ffunc has an optional `ignore_missing` arg. If False (the default), then
any missing values (values=NaN or validity=False) are propagated so that
outputs also have a missing value in any cell that had a missing value in
one of the rows in the fact variable or weight that contributed to that cell.
If `ignore_missing` is True, such input rows are ignored and do not contribute
to the output, much like NumPy's `nanmean` or R's `na.rm = TRUE`. Note this
is also faster and uses less memory.

The `reduce` methods herein all default to the "single NumPy array" format,
with NaN values indicating missingness. Pass e.g. `return_missing_as=(0, False)`
to return a 2-tuple of (values, validity) arrays instead. Functions here will
replace NaN values in the `values` array with 0 in that case. If you prefer
`sum([])` to return 0, for example, without a second "validity" array,
pass `return_missing_as=0`.
"""

import numpy

NaN = float("nan")


def as_separate_validity(arr):
    """Return (values, validity) from the given arr-or-(values, validity)-tuple.

    Most ffuncs take one or more arguments that represent variables with
    missingness; callers have a choice whether to send a single numeric array,
    which uses NaN values to represent missing cells, or a 2-tuple of arrays,
    the first with values and the second with booleans (True meaning "valid"
    and False meaning "missing"). This function returns the 2-tuple form
    regardless of which form the caller passed, because that's more useful
    inside ffunc logic.
    """
    if isinstance(arr, tuple):
        arr, validity = arr
        arr = numpy.asarray(arr)
        validity = numpy.asarray(validity).astype(bool)
    else:
        arr = numpy.asarray(arr)
        validity = ~numpy.isnan(arr)
    return arr, validity


class ffunc:
    """A base class for frequency (and other aggregate) functions."""

    def get_initial_regions(self, cube):
        """Return NumPy arrays to fill, empty except for corner values."""
        raise NotImplementedError

    def fill(self, cube, regions):
        """Fill the `regions` arrays with distributions contingent on cube.dims.

        The given `regions` are assumed to be part of a (possibly larger) cube,
        one which has already been initialized (including any corner values).
        We will compute its common cells from the margins later in self.reduce.
        """
        raise NotImplementedError

    def reduce(self, cube, regions):
        """Return `regions` with common cells calculated and margins removed."""
        raise NotImplementedError

    def calculate(self, cube):
        """Return a NumPy array, the distribution contingent on self.cube.dims."""
        regions = self.get_initial_regions(cube)
        self.fill(cube, regions)
        return self.reduce(cube, regions)

    @staticmethod
    def adjust_zeros(arr, new="nan", condition=None):
        """Set arr[<condition or isclose(arr, 0)>] = new and return it.

        Sometimes, marginal differencing can produce minutely different results
        (within the machine's floating-point epsilon). For most values, this
        has little effect, but when the value should be 0 but is just barely
        not 0, it's easy to end up with e.g. 1e-09/1e-09=1 instead of 0/0=nan.

        Use this to adjust values near zero to "nan" or zero. If `condition`
        is None (the default), then `arr` values near zero will be adjusted
        to the given `new` value. Otherwise, `condition` must be a NumPy
        array, and rows where it is True will cause the corresponding
        rows in `arr` to be zeroed.

        If the given `arr` arg is an array, it is both adjusted in place and
        returned. NumPy scalar values are read-only, so the original cannot
        be adjusted in place, so a new NumPy scalar instance is returned.

        If `arr` is an integer dtype, and any of the conditions hold,
        it will be promoted to float so that NaN can be returned.
        """
        if condition is None:
            condition = numpy.isclose(arr, 0)

        if condition.any():
            if new == "nan" and "i" in arr.dtype.str:
                arr = arr.astype(float)

            if arr.shape:
                arr[condition] = new
            else:
                arr = arr.dtype.type(new)

        return arr


class ffunc_count(ffunc):
    """Calculate the count of a cube.

    If `weights` is given and not None, it must be a NumPy array of numeric
    weight values, or a (weights, validity) tuple, corresponding row-wise
    to any cube.dims.

    If `return_missing_as` is NaN (the default), the `reduce` method will
    return a single numeric NumPy array of counts. Any NaN values in it
    indicate missing cells (an output cell that had no inputs, or a NaN
    weight value, and therefore no count). If `return_missing_as` is a 2-tuple,
    like (0, False), the `reduce` method will return a NumPy array of counts,
    and a second "validity" NumPy array of booleans. Missing values will have
    0 in the former and False in the latter.
    """

    def __init__(
        self, weights=None, N=None, ignore_missing=False, return_missing_as=NaN
    ):
        if weights is None:
            validity = None
        else:
            weights, validity = as_separate_validity(weights)
            weights = weights.copy()
            weights[~validity] = 0

        self.validity = validity
        self.weights = weights
        self.ignore_missing = ignore_missing
        self.return_missing_as = return_missing_as
        if isinstance(self.return_missing_as, tuple):
            self.null = self.return_missing_as[0]
        else:
            self.null = self.return_missing_as
        self.N = N

    def get_initial_regions(self, cube):
        """Return NumPy arrays to fill, empty except for corner values."""
        if self.N is not None:
            N = self.N
        elif cube.dims:
            N = cube.dims[0].shape[0]
        elif self.weights is not None and self.weights.shape:
            N = self.weights.shape[0]
        else:
            raise ValueError(
                "Cannot determine counts with no dimensions, weights, or N."
            )

        if self.weights is None:
            dtype = float if numpy.isnan(self.null) else int
            corner_value = N
        else:
            dtype = float
            if self.weights.shape:
                corner_value = self.weights.sum()
            else:
                corner_value = self.weights * N

        counts = numpy.zeros(cube.working_shape, dtype=dtype)
        # Set the "grand total" value in the corner of each region.
        counts[cube.corner] = corner_value

        if self.weights is None:
            return (counts,)
        else:
            if self.ignore_missing:
                valid_counts = numpy.zeros(cube.working_shape, dtype=int)
                valid_counts[cube.corner] = numpy.count_nonzero(self.validity, axis=0)
                return counts, valid_counts
            else:
                missing_counts = numpy.zeros(cube.working_shape, dtype=int)
                missing_counts[cube.corner] = numpy.count_nonzero(
                    ~self.validity, axis=0
                )
                return counts, missing_counts

    def fill(self, cube, regions):
        """Fill the `regions` arrays with distributions contingent on cube.dims.

        The given `regions` are assumed to be part of a (possibly larger) cube,
        one which has already been initialized (including any corner values).
        We will compute its common cells from the margins later in self.reduce.
        """
        if self.weights is None:
            (counts,) = regions

            def _fill(x_coords, x_rowids):
                counts[x_coords] = len(x_rowids)

        else:
            if self.ignore_missing:
                counts, valid_counts = regions
            else:
                counts, missing_counts = regions

            def _fill(x_coords, x_rowids):
                if self.weights.shape:
                    counts[x_coords] = self.weights[x_rowids].sum()
                    vcount = numpy.count_nonzero(self.validity[x_rowids], axis=0)
                    if self.ignore_missing:
                        valid_counts[x_coords] = vcount
                    else:
                        missing_counts[x_coords] = len(x_rowids) - vcount
                else:
                    # Scalar weight. Broadcast.
                    len_rowids = len(x_rowids)
                    counts[x_coords] = self.weights * len_rowids
                    if self.ignore_missing:
                        valid_counts[x_coords] = len_rowids if self.validity else 0
                    else:
                        missing_counts[x_coords] = 0 if self.validity else len_rowids

        cube.walk(_fill)

    def reduce(self, cube, regions):
        """Return `regions` with common cells calculated and margins removed."""
        if self.weights is None:
            (counts,) = regions

            cube._compute_common_cells_from_marginal_diffs(counts)
            counts = counts[cube.marginless]
            missings = numpy.isclose(counts, 0)
            counts = self.adjust_zeros(counts, new=self.null, condition=missings)

            if isinstance(self.return_missing_as, tuple):
                return counts, ~missings
            else:
                return counts
        else:
            if self.ignore_missing:
                counts, valid_counts = regions
            else:
                counts, missing_counts = regions

            cube._compute_common_cells_from_marginal_diffs(counts)
            counts = counts[cube.marginless]

            if self.ignore_missing:
                cube._compute_common_cells_from_marginal_diffs(valid_counts)
                valid_counts = valid_counts[cube.marginless]
                validity = valid_counts != 0
            else:
                cube._compute_common_cells_from_marginal_diffs(missing_counts)
                missing_counts = missing_counts[cube.marginless]
                validity = missing_counts == 0
            counts = self.adjust_zeros(counts, new=self.null, condition=~validity)

            if isinstance(self.return_missing_as, tuple):
                return counts, validity
            else:
                return counts


class ffunc_valid_count(ffunc):
    """Calculate the valid count of an array contingent on a cube.

    The `arr` arg must be a NumPy array of numeric values to be counted,
    or a tuple of (values, validity) arrays, corresponding row-wise
    to any cube.dims.

    If `weights` is given and not None, it must be a NumPy array of numeric
    weight values, or a (weights, validity) tuple, corresponding row-wise
    to any cube.dims.

    If `return_missing_as` is NaN (the default), the `reduce` method will
    return a single numeric NumPy array of counts. Any NaN values in it
    indicate missing cells (an output cell that had no inputs, or a NaN
    weight value, and therefore no count). If `return_missing_as` is a 2-tuple,
    like (0, False), the `reduce` method will return a NumPy array of counts,
    and a second "validity" NumPy array of booleans. Missing values will have
    0 in the former and False in the latter.
    """

    def __init__(self, arr, weights=None, ignore_missing=False, return_missing_as=NaN):
        _, validity = as_separate_validity(arr)

        if weights is None:
            countables = validity.copy()
        else:
            weights, weights_validity = as_separate_validity(weights)
            validity = (validity.T & weights_validity).T
            countables = (validity.T * weights).T

        countables[~validity] = 0

        self.countables = countables
        self.validity = validity
        self.weights = weights
        self.ignore_missing = ignore_missing
        self.return_missing_as = return_missing_as
        if isinstance(self.return_missing_as, tuple):
            self.null = self.return_missing_as[0]
        else:
            self.null = self.return_missing_as

    def get_initial_regions(self, cube):
        """Return NumPy arrays to fill, empty except for corner values."""
        dtype = int if self.weights is None and not numpy.isnan(self.null) else float

        # countables may itself be an N-dimensional numeric array
        shape = cube.working_shape + self.countables.shape[1:]
        counts = numpy.zeros(shape, dtype=dtype)
        counts[cube.corner] = numpy.nansum(self.countables, axis=0)

        if self.ignore_missing:
            valid_counts = numpy.zeros(shape, dtype=dtype)
            valid_counts[cube.corner] = numpy.count_nonzero(self.validity, axis=0)
            return counts, valid_counts
        else:
            missing_counts = numpy.zeros(shape, dtype=int)
            missing_counts[cube.corner] = numpy.count_nonzero(~self.validity, axis=0)
            return counts, missing_counts

    def fill(self, cube, regions):
        """Fill the `regions` arrays with distributions contingent on cube.dims.

        The given `regions` are assumed to be part of a (possibly larger) cube,
        one which has already been initialized (including any corner values).
        We will compute its common cells from the margins later in self.reduce.
        """
        if self.ignore_missing:
            counts, valid_counts = regions
        else:
            counts, missing_counts = regions

        def _fill(x_coords, x_rowids):
            # This can be called millions of times, so it's critical
            # to perform as few passes over the data as possible.
            # We set countables[~validity] = 0 so there's no need to filter
            # them out again here.
            counts[x_coords] = numpy.sum(self.countables[x_rowids], axis=0)

            vcount = numpy.count_nonzero(self.validity[x_rowids], axis=0)
            if self.ignore_missing:
                valid_counts[x_coords] = vcount
            else:
                missing_counts[x_coords] = len(x_rowids) - vcount

        cube.walk(_fill)

    def reduce(self, cube, regions):
        """Return `regions` with common cells calculated and margins removed."""
        if self.ignore_missing:
            counts, valid_counts = regions
        else:
            counts, missing_counts = regions

        cube._compute_common_cells_from_marginal_diffs(counts)
        counts = counts[cube.marginless]

        if self.ignore_missing:
            cube._compute_common_cells_from_marginal_diffs(valid_counts)
            valid_counts = valid_counts[cube.marginless]
            counts = self.adjust_zeros(counts, self.null, condition=valid_counts == 0)
        else:
            cube._compute_common_cells_from_marginal_diffs(missing_counts)
            missing_counts = missing_counts[cube.marginless]
            counts = self.adjust_zeros(counts, self.null, condition=missing_counts != 0)

        if isinstance(self.return_missing_as, tuple):
            if self.ignore_missing:
                validity = valid_counts != 0
            else:
                validity = missing_counts == 0
            return counts, validity
        else:
            return counts


class ffunc_sum(ffunc):
    """Calculate the sums of an array contingent on a cube.

    The `arr` arg must be a NumPy array of numeric values to be summed,
    or a tuple of (values, validity) arrays, corresponding row-wise
    to any cube.dims.

    If `weights` is given and not None, it must be a NumPy array of numeric
    weight values, or a (weights, validity) tuple, corresponding row-wise
    to any cube.dims.

    If `return_missing_as` is NaN (the default), the `reduce` method will
    return a single numeric NumPy array of sums. Any NaN values in it
    indicate missing cells (an output cell that had no inputs, or a NaN
    weight value, and therefore no sum). If `return_missing_as` is a 2-tuple,
    like (0, False), the `reduce` method will return a NumPy array of sums,
    and a second "validity" NumPy array of booleans. Missing values will have
    0 in the former and False in the latter.
    """

    def __init__(self, arr, weights=None, ignore_missing=False, return_missing_as=NaN):
        summables, validity = as_separate_validity(arr)

        if weights is None:
            summables = summables.copy()
        else:
            weights, weights_validity = as_separate_validity(weights)
            validity = (validity.T & weights_validity).T
            summables = (summables.T * weights).T

        summables[~validity] = 0

        self.summables = summables
        self.validity = validity
        self.weights = weights
        self.ignore_missing = ignore_missing
        self.return_missing_as = return_missing_as
        if isinstance(self.return_missing_as, tuple):
            self.null = self.return_missing_as[0]
        else:
            self.null = self.return_missing_as

    def get_initial_regions(self, cube):
        """Return NumPy arrays to fill, empty except for corner values."""
        dtype = self.summables.dtype if self.weights is None else float

        # summables may itself be an N-dimensional numeric array
        shape = cube.working_shape + self.summables.shape[1:]
        sums = numpy.zeros(shape, dtype=dtype)
        sums[cube.corner] = numpy.nansum(self.summables, axis=0)

        if self.ignore_missing:
            valid_counts = numpy.zeros(shape, dtype=dtype)
            valid_counts[cube.corner] = numpy.count_nonzero(self.validity, axis=0)
            return sums, valid_counts
        else:
            missing_counts = numpy.zeros(shape, dtype=int)
            missing_counts[cube.corner] = numpy.count_nonzero(~self.validity, axis=0)
            return sums, missing_counts

    def fill(self, cube, regions):
        """Fill the `regions` arrays with distributions contingent on cube.dims.

        The given `regions` are assumed to be part of a (possibly larger) cube,
        one which has already been initialized (including any corner values).
        We will compute its common cells from the margins later in self.reduce.
        """
        if self.ignore_missing:
            sums, valid_counts = regions
        else:
            sums, missing_counts = regions

        def _fill(x_coords, x_rowids):
            # This can be called millions of times, so it's critical
            # to perform as few passes over the data as possible.
            # We set summables[~validity] = 0 so there's no need to filter
            # them out again here.
            sums[x_coords] = numpy.sum(self.summables[x_rowids], axis=0)

            vcount = numpy.count_nonzero(self.validity[x_rowids], axis=0)
            if self.ignore_missing:
                valid_counts[x_coords] = vcount
            else:
                missing_counts[x_coords] = len(x_rowids) - vcount

        cube.walk(_fill)

    def reduce(self, cube, regions):
        """Return `regions` with common cells calculated and margins removed."""
        if self.ignore_missing:
            sums, valid_counts = regions
        else:
            sums, missing_counts = regions

        cube._compute_common_cells_from_marginal_diffs(sums)
        sums = sums[cube.marginless]

        if self.ignore_missing:
            cube._compute_common_cells_from_marginal_diffs(valid_counts)
            valid_counts = valid_counts[cube.marginless]
            sums = self.adjust_zeros(sums, self.null, condition=valid_counts == 0)
        else:
            cube._compute_common_cells_from_marginal_diffs(missing_counts)
            missing_counts = missing_counts[cube.marginless]
            sums = self.adjust_zeros(sums, self.null, condition=missing_counts != 0)

        if isinstance(self.return_missing_as, tuple):
            if self.ignore_missing:
                validity = valid_counts != 0
            else:
                validity = missing_counts == 0
            return sums, validity
        else:
            return sums


class ffunc_mean(ffunc):
    """Calculate the means of an array contingent on a cube.

    The `arr` arg must be a NumPy array of numeric values to be meaned,
    or a tuple of (values, validity) arrays, corresponding row-wise
    to any cube.dims.

    If `weights` is given and not None, it must be a NumPy array of numeric
    weight values, or a (weights, validity) tuple, corresponding row-wise
    to any cube.dims.

    If `return_missing_as` is NaN (the default), the `reduce` method will
    return a single numeric NumPy array of means. Any NaN values in it
    indicate missing cells (an output cell that had no inputs, or a NaN
    weight value, and therefore no mean). If `return_missing_as` is a 2-tuple,
    like (0, False), the `reduce` method will return a NumPy array of means,
    and a second "validity" NumPy array of booleans. Missing values will have
    0 in the former and False in the latter.
    """

    def __init__(self, arr, weights=None, ignore_missing=False, return_missing_as=NaN):
        summables, validity = as_separate_validity(arr)

        if weights is None:
            summables = summables.copy()
            countables = validity.astype(int)
        else:
            weights, weights_validity = as_separate_validity(weights)
            validity = (validity.T & weights_validity).T
            summables = (summables.T * weights).T
            countables = (validity.T * weights).T

        summables[~validity] = 0
        countables[~validity] = 0

        self.summables = summables
        self.validity = validity
        self.countables = countables
        self.weights = weights
        self.ignore_missing = ignore_missing
        self.return_missing_as = return_missing_as
        if isinstance(self.return_missing_as, tuple):
            self.null = self.return_missing_as[0]
        else:
            self.null = self.return_missing_as

    def get_initial_regions(self, cube):
        """Return NumPy arrays to fill, empty except for corner values."""
        dtype = int if self.weights is None else float

        # summables may itself be an N-dimensional numeric array
        shape = cube.working_shape + self.summables.shape[1:]
        sums = numpy.zeros(shape, dtype=self.summables.dtype)
        valid_counts = numpy.zeros(shape, dtype=dtype)

        # Set the "grand total" value in the corner of each region.
        sums[cube.corner] = numpy.nansum(self.summables, axis=0)
        valid_counts[cube.corner] = numpy.sum(self.countables, axis=0)

        if self.ignore_missing:
            return sums, valid_counts
        else:
            # Keep a third array for marking which output cells
            # have a missing value in the fact variable or a weight.
            # Unlike valid_counts, which uses self.countables which is weighted,
            # this uses self.validity which is unweighted.
            missing_counts = numpy.zeros(shape, dtype=int)
            missing_counts[cube.corner] = numpy.count_nonzero(~self.validity, axis=0)
            return sums, valid_counts, missing_counts

    def fill(self, cube, regions):
        """Fill the `regions` arrays with distributions contingent on cube.dims.

        The given `regions` are assumed to be part of a (possibly larger) cube,
        one which has already been initialized (including any corner values).
        We will compute its common cells from the margins later in self.reduce.
        """
        if self.ignore_missing:
            sums, valid_counts = regions
        else:
            sums, valid_counts, missing_counts = regions

        def _fill(x_coords, x_rowids):
            # This can be called millions of times, so it's critical
            # to perform as few passes over the data as possible.
            # We set summables/countables[~validity] = 0 so there's
            # no need to filter them out again here.
            sums[x_coords] = numpy.nansum(self.summables[x_rowids], axis=0, dtype=float)
            valid_counts[x_coords] = numpy.sum(self.countables[x_rowids], axis=0)
            if not self.ignore_missing:
                vcount = numpy.count_nonzero(self.validity[x_rowids], axis=0)
                missing_counts[x_coords] = len(x_rowids) - vcount

        cube.walk(_fill)

    def reduce(self, cube, regions):
        """Return `regions` with common cells calculated and margins removed."""
        if self.ignore_missing:
            sums, valid_counts = regions
        else:
            sums, valid_counts, missing_counts = regions

        cube._compute_common_cells_from_marginal_diffs(sums)
        cube._compute_common_cells_from_marginal_diffs(valid_counts)

        sums = sums[cube.marginless]
        valid_counts = valid_counts[cube.marginless]
        valid_counts = self.adjust_zeros(valid_counts, new=0)

        with numpy.errstate(divide="ignore", invalid="ignore"):
            means = sums / valid_counts

        if self.ignore_missing:
            means = self.adjust_zeros(means, self.null, condition=valid_counts == 0)
        else:
            cube._compute_common_cells_from_marginal_diffs(missing_counts)
            missing_counts = missing_counts[cube.marginless]
            means = self.adjust_zeros(means, self.null, condition=missing_counts != 0)

        if isinstance(self.return_missing_as, tuple):
            if self.ignore_missing:
                validity = valid_counts != 0
            else:
                validity = missing_counts == 0
            return means, validity
        else:
            return means
