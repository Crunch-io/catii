"""Frequency (and other aggregate) functions for catii.

These take xcubes as input and return NumPy arrays as output from their `reduce`
methods, In between, they create, fill, and reduce "regions" (NumPy arrays)
as intermediate workspaces. The number of regions and their dtype depends on the
function: xfunc_count, for example, uses one region of ints or floats and mostly
returns it unchanged, while xfunc_mean uses separate "sums" and "valid_counts"
regions as the numerator and denominator, dividing them in its "reduce" method
to return a single array of means.

Most arguments to xfuncs (including weights) are variables which correspond
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

Each xfunc has an optional `ignore_missing` arg. If False (the default), then
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

    Most xfuncs take one or more arguments that represent variables with
    missingness; callers have a choice whether to send a single numeric array,
    which uses NaN values to represent missing cells, or a 2-tuple of arrays,
    the first with values and the second with booleans (True meaning "valid"
    and False meaning "missing"). This function returns the 2-tuple form
    regardless of which form the caller passed, because that's more useful
    inside xfunc logic.
    """
    if isinstance(arr, tuple):
        arr, validity = arr
        arr = numpy.asarray(arr)
        validity = numpy.asarray(validity).astype(bool)
    else:
        arr = numpy.asarray(arr)
        validity = ~numpy.isnan(arr)
    return arr, validity


class xfunc:
    """A base class for frequency (and other aggregate) functions."""

    shape = None
    """The shape of any fact variable(s), not including its N; that is,
    arr.shape[1:], and represents any additional axes the fact variable has.
    If non-empty, the `reduce` output will typically by extended by these
    additional axes, as well.
    """

    def get_initial_regions(self, cube):
        """Return empty NumPy arrays to fill."""
        raise NotImplementedError

    def flat_regions(self, regions):
        """Return the given regions, flattened except for self.shape."""
        if self.size:
            return [
                part.reshape((int(part.size / self.size),) + self.shape)
                for part in regions
            ]
        else:
            return [part.reshape((part.size,) + self.shape) for part in regions]

    def fill(self, cube, regions):
        """Fill the `regions` arrays with distributions contingent on cube.dims.

        The given `regions` are assumed to be part of a (possibly larger) cube,
        one which has already been initialized (including any corner values).
        We will compute its common cells from the margins later in self.reduce.
        """
        raise NotImplementedError

    def reduce(self, cube, regions):
        """Return `regions` reduced to proper output."""
        raise NotImplementedError

    def calculate(self, cube):
        """Return a NumPy array, the distribution contingent on self.cube.dims."""
        regions = self.get_initial_regions(cube)
        self.fill(cube, regions)
        return self.reduce(cube, regions)

    @staticmethod
    def adjust_zeros(arr, new="nan", condition=None):
        """Set arr[<condition or isclose(arr, 0)>] = new and return it.

        Use this to adjust values to "nan" or zero. If `condition`
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

    @staticmethod
    def bins(coordinates):
        """Yield a bin number and boolean row mask for each result cell."""
        # TODO: When the number of distinct values grows large, it might be
        # faster to use argsort and form N slices of the output (with
        # lengths found by bincount, then cumulative sum to find offsets)
        # rather than perform N (row_indexes == i) passes.
        uniqs, row_indexes = numpy.unique(coordinates, return_inverse=True)
        for i, u in enumerate(uniqs):
            yield u, (row_indexes == i)


class xfunc_count(xfunc):
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
        self.shape = () if validity is None else validity.shape[1:]
        self.size = numpy.prod(self.shape, 0)

        self.weights = weights
        self.ignore_missing = ignore_missing
        self.return_missing_as = return_missing_as
        if isinstance(self.return_missing_as, tuple):
            self.null = self.return_missing_as[0]
        else:
            self.null = self.return_missing_as
        if N is None and self.weights is not None and self.weights.shape:
            N = self.weights.shape[0]
        self.N = N

    def get_initial_regions(self, cube):
        """Return empty NumPy arrays to fill."""
        dtype = int if self.weights is None and not numpy.isnan(self.null) else float

        shape = cube.shape
        if not shape:
            shape = (1,)
        counts = numpy.zeros(shape, dtype=dtype)

        if self.weights is None:
            return (counts,)
        else:
            if self.ignore_missing:
                valid_counts = numpy.zeros(shape, dtype=int)
                return counts, valid_counts
            else:
                missing_counts = numpy.zeros(shape, dtype=int)
                return counts, missing_counts

    def fill(self, coordinates, regions):
        """Fill the `regions` arrays with distributions contingent on coordinates.

        The given `regions` are assumed to be part of a (possibly larger) cube,
        one which has already been initialized (including any corner values).
        We will compute its common cells from the margins later in self.reduce.
        """
        # Flatten our regions so flat bincount output can overwrite it in place.
        regions = self.flat_regions(regions)

        if self.weights is None:
            (counts,) = regions
            if coordinates is None:
                # Dimensionless cube
                if self.N is None:
                    raise ValueError(
                        "Cannot determine counts with no dimensions, weights, or N."
                    )
                counts[:] = self.N
            else:
                counts[:] = numpy.bincount(coordinates, minlength=counts.shape[0])
        else:
            if self.ignore_missing:
                counts, valid_counts = regions
            else:
                counts, missing_counts = regions

            if coordinates is None:
                # Dimensionless cube
                if self.N is None:
                    raise ValueError(
                        "Cannot determine counts with no dimensions, weights, or N."
                    )
                counts[:] = self.weights.sum()
            else:
                size = counts.shape[0]
                if self.weights.shape:
                    counts[:] = numpy.bincount(
                        coordinates, weights=self.weights, minlength=size
                    )
                    if self.ignore_missing:
                        valid_counts[:] = numpy.bincount(
                            coordinates, weights=self.validity, minlength=size
                        )
                    else:
                        missing_counts[:] = numpy.bincount(
                            coordinates, weights=~self.validity, minlength=size
                        )
                else:
                    bcounts = numpy.bincount(coordinates, minlength=size)
                    counts[:] = bcounts * self.weights
                    if self.ignore_missing:
                        valid_counts[:] = bcounts * self.validity
                    else:
                        missing_counts[:] = bcounts * ~self.validity

    def reduce(self, cube, regions):
        """Return `regions` reduced to proper output."""
        if self.weights is None:
            (counts,) = regions
            missings = numpy.isclose(counts, 0)
            counts = self.adjust_zeros(counts, self.null, condition=missings)

            if isinstance(self.return_missing_as, tuple):
                return counts, ~missings
            else:
                return counts
        else:
            if self.ignore_missing:
                counts, valid_counts = regions
                counts = self.adjust_zeros(
                    counts, self.null, condition=valid_counts == 0
                )
            else:
                counts, missing_counts = regions
                counts = self.adjust_zeros(
                    counts, self.null, condition=missing_counts != 0
                )

            if isinstance(self.return_missing_as, tuple):
                if self.ignore_missing:
                    validity = valid_counts != 0
                else:
                    validity = missing_counts == 0
                return counts, validity
            else:
                return counts


class xfunc_valid_count(xfunc):
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
        self.shape = validity.shape[1:]
        self.size = numpy.prod(self.shape, 0)

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
        """Return empty NumPy arrays to fill."""
        dtype = int if self.weights is None and not numpy.isnan(self.null) else float

        # countables may itself be an N-dimensional numeric array
        shape = cube.shape + self.countables.shape[1:]
        if not shape:
            shape = (1,)
        counts = numpy.zeros(shape, dtype=dtype)

        if self.ignore_missing:
            valid_counts = numpy.zeros(shape, dtype=int)
            return counts, valid_counts
        else:
            missing_counts = numpy.zeros(shape, dtype=int)
            return counts, missing_counts

    def fill(self, coordinates, regions):
        """Fill the `regions` arrays with distributions contingent on coordinates.

        The given `regions` are assumed to be part of a (possibly larger) cube,
        one which has already been initialized (including any corner values).
        We will compute its common cells from the margins later in self.reduce.
        """
        # Flatten our regions so flat bincount output can overwrite it in place.
        regions = self.flat_regions(regions)

        if self.ignore_missing:
            counts, valid_counts = regions
        else:
            counts, missing_counts = regions

        # This can be called thousands of times, so it's critical
        # to perform as few passes over the data as possible.
        # We set countables[~validity] = 0 so there's
        # no need to filter them out again here.
        if coordinates is None:
            counts[:] = numpy.nansum(self.countables, axis=0)
            if self.ignore_missing:
                valid_counts[:] = numpy.count_nonzero(self.validity, axis=0)
            else:
                missing_counts[:] = numpy.count_nonzero(~self.validity, axis=0)
        else:
            size = counts.shape[0]
            if self.countables.ndim == 1:
                counts[:] = numpy.bincount(
                    coordinates, weights=self.countables, minlength=size
                )
                if self.ignore_missing:
                    valid_counts[:] = numpy.bincount(
                        coordinates, weights=self.validity, minlength=size
                    )
                else:
                    missing_counts[:] = numpy.bincount(
                        coordinates, weights=~self.validity, minlength=size
                    )
            elif self.countables.ndim == 2:
                for i, ss in enumerate(self.countables.T):
                    counts[:, i] = numpy.bincount(
                        coordinates, weights=ss, minlength=size
                    )
                if self.ignore_missing:
                    for i, cs in enumerate(self.validity.T):
                        valid_counts[:, i] = numpy.bincount(
                            coordinates, weights=cs, minlength=size
                        )
                else:
                    for i, cs in enumerate(~self.validity.T):
                        missing_counts[:, i] = numpy.bincount(
                            coordinates, weights=cs, minlength=size
                        )

    def reduce(self, cube, regions):
        """Return `regions` reduced to proper output."""
        if self.ignore_missing:
            counts, valid_counts = regions
            counts = self.adjust_zeros(counts, self.null, condition=valid_counts == 0)
        else:
            counts, missing_counts = regions
            counts = self.adjust_zeros(counts, self.null, condition=missing_counts != 0)

        if isinstance(self.return_missing_as, tuple):
            if self.ignore_missing:
                validity = valid_counts != 0
            else:
                validity = missing_counts == 0
            return counts, validity
        else:
            return counts


class xfunc_sum(xfunc):
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
        self.shape = summables.shape[1:]
        self.size = numpy.prod(self.shape, 0)

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
        """Return empty NumPy arrays to fill."""
        dtype = self.summables.dtype if self.weights is None else float

        # summables may itself be an N-dimensional numeric array
        shape = cube.shape + self.summables.shape[1:]
        if not shape:
            shape = (1,)
        sums = numpy.zeros(shape, dtype=dtype)

        if self.ignore_missing:
            valid_counts = numpy.zeros(shape, dtype=int)
            return sums, valid_counts
        else:
            missing_counts = numpy.zeros(shape, dtype=int)
            return sums, missing_counts

    def fill(self, coordinates, regions):
        """Fill the `regions` arrays with distributions contingent on coordinates.

        The given `regions` are assumed to be part of a (possibly larger) cube,
        one which has already been initialized (including any corner values).
        We will compute its common cells from the margins later in self.reduce.
        """
        # Flatten our regions so flat bincount output can overwrite it in place.
        regions = self.flat_regions(regions)

        if self.ignore_missing:
            sums, valid_counts = regions
        else:
            sums, missing_counts = regions

        # This can be called thousands of times, so it's critical
        # to perform as few passes over the data as possible.
        # We set summables/countables[~validity] = 0 so there's
        # no need to filter them out again here.

        if coordinates is None:
            sums[:] = numpy.nansum(self.summables, axis=0)
            if self.ignore_missing:
                valid_counts[:] = numpy.count_nonzero(self.validity, axis=0)
            else:
                missing_counts[:] = numpy.count_nonzero(~self.validity, axis=0)
        else:
            size = sums.shape[0]
            if self.summables.ndim == 1:
                sums[:] = numpy.bincount(
                    coordinates, weights=self.summables, minlength=size
                )
                if self.ignore_missing:
                    valid_counts[:] = numpy.bincount(
                        coordinates, weights=self.validity, minlength=size
                    )
                else:
                    missing_counts[:] = numpy.bincount(
                        coordinates, weights=~self.validity, minlength=size
                    )
            elif self.summables.ndim == 2:
                for i, ss in enumerate(self.summables.T):
                    sums[:, i] = numpy.bincount(coordinates, weights=ss, minlength=size)
                if self.ignore_missing:
                    for i, cs in enumerate(self.validity.T):
                        valid_counts[:, i] = numpy.bincount(
                            coordinates, weights=cs, minlength=size
                        )
                else:
                    for i, cs in enumerate(~self.validity.T):
                        missing_counts[:, i] = numpy.bincount(
                            coordinates, weights=cs, minlength=size
                        )

    def reduce(self, cube, regions):
        """Return `regions` reduced to proper output."""
        if self.ignore_missing:
            sums, valid_counts = regions
            sums = self.adjust_zeros(sums, self.null, condition=valid_counts == 0)
        else:
            sums, missing_counts = regions
            sums = self.adjust_zeros(sums, self.null, condition=missing_counts != 0)

        if isinstance(self.return_missing_as, tuple):
            if self.ignore_missing:
                validity = valid_counts != 0
            else:
                validity = missing_counts == 0
            return sums, validity
        else:
            return sums


class xfunc_mean(xfunc):
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
        self.shape = summables.shape[1:]
        self.size = numpy.prod(self.shape, 0)

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
        """Return empty NumPy arrays to fill."""
        # summables may itself be an N-dimensional numeric array
        shape = cube.shape + self.summables.shape[1:]
        if not shape:
            shape = (1,)
        sums = numpy.zeros(shape, dtype=float)
        # This is *weighted* counts.
        valid_counts = numpy.zeros(shape, dtype=float)
        if self.ignore_missing:
            return sums, valid_counts
        else:
            # Keep a third array for marking which output cells
            # have a missing value in the fact variable or a weight.
            # Unlike valid_counts, which uses self.countables which is weighted,
            # this uses self.validity which is unweighted.
            missing_counts = numpy.zeros(shape, dtype=int)
            return sums, valid_counts, missing_counts

    def fill(self, coordinates, regions):
        """Fill the `regions` arrays with distributions contingent on coordinates.

        The given `regions` are assumed to be part of a (possibly larger) cube,
        one which has already been initialized (including any corner values).
        We will compute its common cells from the margins later in self.reduce.
        """
        # Flatten our regions so flat bincount output can overwrite it in place.
        regions = self.flat_regions(regions)

        if self.ignore_missing:
            sums, valid_counts = regions
        else:
            sums, valid_counts, missing_counts = regions

        # This can be called thousands of times, so it's critical
        # to perform as few passes over the data as possible.
        # We set summables/countables[~validity] = 0 so there's
        # no need to filter them out again here.

        if coordinates is None:
            sums[:] = numpy.nansum(self.summables, axis=0)
            valid_counts[:] = numpy.sum(self.countables, axis=0)
            if not self.ignore_missing:
                missing_counts[:] = numpy.sum(~self.validity, axis=0)
        else:
            size = sums.shape[0]
            if self.summables.ndim == 1:
                sums[:] = numpy.bincount(
                    coordinates, weights=self.summables, minlength=size
                )
                valid_counts[:] = numpy.bincount(
                    coordinates, weights=self.countables, minlength=size
                )
            elif self.summables.ndim == 2:
                for i, ss in enumerate(self.summables.T):
                    sums[:, i] = numpy.bincount(coordinates, weights=ss, minlength=size)
                for i, cs in enumerate(self.countables.T):
                    valid_counts[:, i] = numpy.bincount(
                        coordinates, weights=cs, minlength=size
                    )
            if not self.ignore_missing:
                missing_counts[:] = numpy.bincount(
                    coordinates, weights=~self.validity, minlength=size
                )

    def reduce(self, cube, regions):
        """Return `regions` reduced to proper output."""
        if self.ignore_missing:
            sums, valid_counts = regions
            with numpy.errstate(divide="ignore", invalid="ignore"):
                means = sums / valid_counts
            means = self.adjust_zeros(means, self.null, condition=valid_counts == 0)
        else:
            sums, valid_counts, missing_counts = regions
            with numpy.errstate(divide="ignore", invalid="ignore"):
                means = sums / valid_counts
            means = self.adjust_zeros(means, self.null, condition=missing_counts != 0)

        if isinstance(self.return_missing_as, tuple):
            if self.ignore_missing:
                validity = valid_counts != 0
            else:
                validity = missing_counts == 0
            return means, validity
        else:
            return means


class xfunc_stddev(xfunc):
    """Calculate the standard deviations of an array contingent on a cube.

    The `arr` arg must be a NumPy array of numeric values to be stddeved,
    or a tuple of (values, validity) arrays, corresponding row-wise
    to any cube.dims.

    If `weights` is given and not None, it must be a NumPy array of numeric
    weight values, or a (weights, validity) tuple, corresponding row-wise
    to any cube.dims.

    If `return_missing_as` is NaN (the default), the `reduce` method will
    return a single numeric NumPy array of stddevs. Any NaN values in it
    indicate missing cells (an output cell that had no inputs, or a NaN
    weight value, and therefore no stddev). If `return_missing_as` is a 2-tuple,
    like (0, False), the `reduce` method will return a NumPy array of stddevs,
    and a second "validity" NumPy array of booleans. Missing values will have
    0 in the former and False in the latter.
    """

    def __init__(self, arr, weights=None, ignore_missing=False, return_missing_as=NaN):
        summables, validity = as_separate_validity(arr)
        self.shape = summables.shape[1:]
        self.size = numpy.prod(self.shape, 0)

        if weights is None:
            countables = validity.astype(int)
        else:
            weights, weights_validity = as_separate_validity(weights)
            validity = (validity.T & weights_validity).T
            countables = (validity.T * weights).T

        summables = summables.astype(float)
        summables = summables.copy()
        summables[~validity] = float("nan")
        if weights is None:
            self.wsummables = summables
        else:
            self.wsummables = (summables.T * weights).T

        # This is *weighted* counts.
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
        """Return empty NumPy arrays to fill."""
        # summables may itself be an N-dimensional numeric array
        shape = cube.shape + self.summables.shape[1:]
        if not shape:
            shape = (1,)
        stddevs = numpy.full(shape, self.null, dtype=float)
        if self.ignore_missing:
            valid_counts = numpy.zeros(shape, dtype=int)
            return stddevs, valid_counts
        else:
            missing_counts = numpy.zeros(shape, dtype=int)
            return stddevs, missing_counts

    def fill(self, coordinates, regions):
        """Fill the `regions` arrays with distributions contingent on coordinates.

        The given `regions` are assumed to be part of a (possibly larger) cube,
        one which has already been initialized (including any corner values).
        We will compute its common cells from the margins later in self.reduce.
        """
        # Flatten our regions so flat bincount output can overwrite it in place.
        regions = self.flat_regions(regions)

        if coordinates is None:
            if self.summables.ndim == 1:
                self._fill_one_no_coordinates(
                    regions,
                    self.summables,
                    self.wsummables,
                    self.countables,
                    self.validity,
                )
            elif self.summables.ndim == 2:
                for i in range(self.summables.shape[1]):
                    self._fill_one_no_coordinates(
                        [r[:, i] for r in regions],
                        self.summables[:, i],
                        self.wsummables[:, i],
                        self.countables[:, i],
                        self.validity[:, i],
                    )
        else:
            if self.summables.ndim == 1:
                self._fill_one_by_coordinates(
                    regions,
                    coordinates,
                    self.summables,
                    self.wsummables,
                    self.countables,
                    self.validity,
                )
            elif self.summables.ndim == 2:
                for i in range(self.summables.shape[1]):
                    self._fill_one_by_coordinates(
                        [r[:, i] for r in regions],
                        coordinates,
                        self.summables[:, i],
                        self.wsummables[:, i],
                        self.countables[:, i],
                        self.validity[:, i],
                    )

    def _fill_one_no_coordinates(
        self, regions, summables, wsummables, countables, validity
    ):
        # Weighted mean
        if self.ignore_missing:
            stddevs, valid_counts = regions
            summables = summables[validity]
            wsummables = wsummables[validity]
            countables = countables[validity]
            weights = None if self.weights is None else self.weights[validity]
        else:
            stddevs, missing_counts = regions
            weights = self.weights

        N = numpy.count_nonzero(validity, axis=0)
        if len(summables) >= 2:
            # Weighted mean
            wsums = numpy.nansum(wsummables, axis=0)
            wcounts = numpy.nansum(countables, axis=0)
            with numpy.errstate(divide="ignore", invalid="ignore"):
                wmeans = wsums / wcounts

            # Note we subtract *weighted* means from *UN-weighted* values.
            squared_variances = (summables - wmeans) ** 2
            if weights is not None:
                squared_variances = (squared_variances.T * weights).T

            varsums = numpy.nansum(squared_variances, axis=0)

            with numpy.errstate(divide="ignore", invalid="ignore"):
                if weights is None:
                    stddevs[:] = numpy.sqrt(varsums / (N - 1))
                else:
                    stddevs[:] = numpy.sqrt(
                        (varsums / numpy.nansum(weights)) * (N / (N - 1))
                    )

        if self.ignore_missing:
            valid_counts[:] = N
        else:
            missing_counts[:] = len(self.summables) - N

    def _fill_one_by_coordinates(
        self, regions, coordinates, summables, wsummables, countables, validity
    ):
        if self.ignore_missing:
            stddevs, valid_counts = regions
            coords = coordinates[validity]
            summables = summables[validity]
            wsummables = wsummables[validity]
            countables = countables[validity]
            weights = None if self.weights is None else self.weights[validity]
        else:
            stddevs, missing_counts = regions
            coords = coordinates
            weights = self.weights
        size = stddevs.shape[0]

        # Weighted mean
        wsums = numpy.bincount(coords, weights=wsummables, minlength=size)
        wcounts = numpy.bincount(coords, weights=countables, minlength=size)
        with numpy.errstate(divide="ignore", invalid="ignore"):
            wmeans = wsums / wcounts

        # Note we subtract *weighted* means from *UN-weighted* values.
        # Neat trick: wmeans[coords] gives us the appropriate
        # classified mean for each input row.
        squared_variances = (summables - wmeans[coords]) ** 2
        if weights is not None:
            squared_variances = (squared_variances.T * weights).T
        varsums = numpy.bincount(coords, weights=squared_variances, minlength=size)
        N = numpy.bincount(coords, minlength=size)

        with numpy.errstate(divide="ignore", invalid="ignore"):
            if weights is None:
                stddevs[:] = numpy.sqrt(varsums / (N - 1))
            else:
                weightsums = numpy.bincount(coords, weights=weights, minlength=size)
                stddevs[:] = numpy.sqrt((varsums / weightsums) * (N / (N - 1)))

        if self.ignore_missing:
            valid_counts[:] = N
        else:
            missing_counts[:] = numpy.bincount(
                coordinates, weights=~validity, minlength=size
            )

    def reduce(self, cube, regions):
        """Return `regions` reduced to proper output."""
        if self.ignore_missing:
            stddevs, valid_counts = regions
            stddevs = self.adjust_zeros(stddevs, self.null, condition=valid_counts < 2)
        else:
            stddevs, missing_counts = regions
            stddevs = self.adjust_zeros(
                stddevs, self.null, condition=missing_counts != 0
            )

        if isinstance(self.return_missing_as, tuple):
            if self.ignore_missing:
                validity = valid_counts > 1
            else:
                validity = missing_counts == 0
            return stddevs, validity
        else:
            return stddevs


class xfunc_quantile(xfunc):
    """Return the q-quantile of an array contingent on a cube.

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

    def __init__(
        self,
        arr,
        probability,
        weights=None,
        ignore_missing=False,
        return_missing_as=NaN,
    ):
        # Coerce to a float array because of this gem:
        # >>> numpy.nanquantile([1.0, float("inf")], 0)
        # 1.0
        # >>> numpy.nanquantile([1.0, float("inf")], 0.0)
        # nan
        self.probability = numpy.array(probability, dtype=float)

        arr, validity = as_separate_validity(arr)
        self.shape = arr.shape[1:]
        self.size = numpy.prod(self.shape, 0)

        if weights is None:
            arr = arr.copy()
        else:
            weights, weights_validity = as_separate_validity(weights)
            validity = (validity.T & weights_validity).T

        arr[~validity] = NaN

        self.arr = arr

        # Set negative weights to 0, like SAS EXCLNPWGT option
        if weights is not None:
            neg_weights = weights < 0
            if neg_weights.any():
                weights = weights.copy()
                weights[neg_weights] = 0
        self.weights = weights

        self.ignore_missing = ignore_missing
        if self.ignore_missing:
            self.qfunc = numpy.nanquantile
        else:
            self.qfunc = numpy.quantile

        self.return_missing_as = return_missing_as
        if isinstance(self.return_missing_as, tuple):
            self.null = self.return_missing_as[0]
        else:
            self.null = self.return_missing_as

    def get_initial_regions(self, cube):
        """Return empty NumPy arrays to fill."""
        shape = cube.shape + self.shape
        if not shape:
            shape = (1,)
        qs = numpy.full(shape, self.null, dtype=float)
        return (qs,)

    def fill(self, coordinates, regions):
        """Fill the `regions` arrays with distributions contingent on coordinates.

        The given `regions` are assumed to be part of a (possibly larger) cube,
        one which has already been initialized (including any corner values).
        We will compute its common cells from the margins later in self.reduce.
        """
        (qs,) = self.flat_regions(regions)

        if self.weights is None:
            if coordinates is None:
                qs[:] = self.qfunc(self.arr, self.probability, axis=0)
            else:
                for i, rowmask in self.bins(coordinates):
                    qs[i] = self.qfunc(self.arr[rowmask], self.probability, axis=0)
        else:
            if coordinates is None:
                qs[:] = self.weighted_quantile(self.arr, self.probability, self.weights)
            else:
                for i, rowmask in self.bins(coordinates):
                    if self.weights.shape:
                        w = self.weights[rowmask]
                    else:
                        w = numpy.repeat(self.weights, len(rowmask))
                    qs[i] = self.weighted_quantile(
                        self.arr[rowmask], self.probability, w
                    )

    def weighted_quantile(self, arr, probability, weights):
        def weighted_quantile_1d(a):
            w = weights

            ind = a.argsort()
            a = a[ind]
            w = w[ind]

            if self.ignore_missing:
                missing = numpy.isnan(a) | numpy.isnan(w)
                if numpy.any(missing):
                    valid = ~missing
                    a = a[valid]
                    w = w[valid]

            N = len(w)
            if N == 0:
                return NaN

            cs = w.cumsum(axis=0)
            prob = probability * cs[-1]
            right = numpy.digitize(prob, cs)
            left = (right - 1).clip(min=0)
            xdiff = numpy.diff(a, append=[0], axis=0)
            with numpy.errstate(divide="ignore", invalid="ignore"):
                frac = (prob - cs[left]).clip(min=0) / w[right.clip(max=len(w) - 1)]
                return a[left] + frac * xdiff[left]

        return numpy.apply_along_axis(weighted_quantile_1d, 0, arr)

    def reduce(self, cube, regions):
        """Return `regions` reduced to proper output."""
        (qs,) = regions
        if isinstance(self.return_missing_as, tuple):
            missings = numpy.isnan(qs)
            qs[missings] = self.null
            return qs, ~missings
        else:
            return qs


class xfunc_op_base(xfunc):
    """Calculate self.op() of an array contingent on a cube.

    The `arr` arg must be a NumPy array of values, or a tuple of
    (values, validity) arrays, corresponding row-wise to any cube.dims.

    If `return_missing_as` is NaN (the default), the `reduce` method will
    return a single numeric NumPy array of values. Any NaN values in it
    indicate missing cells (an output cell that had no inputs, and therefore
    no value). If `return_missing_as` is a 2-tuple, like (0, False),
    the `reduce` method will return a NumPy array of values, and a second
    "validity" NumPy array of booleans. Missing values will have 0 in the
    former and False in the latter.
    """

    def __init__(self, arr, ignore_missing=False, return_missing_as=NaN):
        values, validity = as_separate_validity(arr)
        self.shape = values.shape[1:]
        self.size = numpy.prod(self.shape, 0)

        self.values = values
        self.validity = validity
        self.ignore_missing = ignore_missing
        self.return_missing_as = return_missing_as
        if isinstance(self.return_missing_as, tuple):
            self.null = self.return_missing_as[0]
        else:
            self.null = self.return_missing_as

    def get_initial_regions(self, cube):
        """Return empty NumPy arrays to fill."""
        # values may itself be an N-dimensional numeric array
        shape = cube.shape + self.values.shape[1:]
        if not shape:
            shape = (1,)

        dtype = self.values.dtype
        if dtype is not float:
            try:
                if numpy.isnan(self.null):
                    dtype = float
            except TypeError:
                pass

        output_values = numpy.full(shape, self.null, dtype=dtype)
        output_validity = numpy.zeros(shape, dtype=bool)
        return output_values, output_validity

    def fill(self, coordinates, regions):
        """Fill the `regions` arrays with distributions contingent on coordinates.

        The given `regions` are assumed to be part of a (possibly larger) cube,
        one which has already been initialized (including any corner values).
        We will compute its common cells from the margins later in self.reduce.
        """
        output_values, output_validity = self.flat_regions(regions)

        if coordinates is None:
            # We had no coordinates because there was no grouping defined.
            if self.ignore_missing:
                values = self.values[self.validity]
                if len(values):
                    output_values[:] = self.op(values, axis=0)
                    output_validity[:] = True
            else:
                if len(self.values):
                    if numpy.all(self.validity):
                        output_values[:] = self.op(self.values, axis=0)
                        output_validity[:] = True
        else:
            values = self.values
            if self.ignore_missing:
                values = values[self.validity]
                coordinates = coordinates[self.validity]

            for i in range(output_values.shape[0]):
                matches = values[coordinates == i]
                if len(matches):
                    output_values[i] = self.op(matches, axis=0)
                    output_validity[i] = True

    def reduce(self, cube, regions):
        """Return `regions` reduced to proper output."""
        output_values, output_validity = regions
        output_values[~output_validity] = self.null

        if isinstance(self.return_missing_as, tuple):
            return output_values, output_validity
        else:
            return output_values


class xfunc_max(xfunc_op_base):
    """Calculate the max of an array contingent on a cube.

    The `arr` arg must be a NumPy array of values, or a tuple of
    (values, validity) arrays, corresponding row-wise to any cube.dims.

    If `return_missing_as` is NaN (the default), the `reduce` method will
    return a single numeric NumPy array of maximums. Any NaN values in it
    indicate missing cells (an output cell that had no inputs, and
    therefore no max). If `return_missing_as` is a 2-tuple, like (0, False),
    the `reduce` method will return a NumPy array of maximums, and a second
    "validity" NumPy array of booleans. Missing values will have 0 in the
    former and False in the latter.
    """

    op = staticmethod(numpy.amax)


class xfunc_min(xfunc_op_base):
    """Calculate the min of an array contingent on a cube.

    The `arr` arg must be a NumPy array of values, or a tuple of
    (values, validity) arrays, corresponding row-wise to any cube.dims.

    If `return_missing_as` is NaN (the default), the `reduce` method will
    return a single numeric NumPy array of minimums. Any NaN values in it
    indicate missing cells (an output cell that had no inputs, and
    therefore no min). If `return_missing_as` is a 2-tuple, like (0, False),
    the `reduce` method will return a NumPy array of minimums, and a second
    "validity" NumPy array of booleans. Missing values will have 0 in the
    former and False in the latter.
    """

    op = staticmethod(numpy.amin)


class xfunc_corrcoef(xfunc):
    """Calculate the correlation coefficients of an array contingent on a cube.

    The `arr` arg must be a NumPy array of numeric values to be analyzed,
    or a tuple of (values, validity) arrays, corresponding row-wise
    to any cube.dims.

    If `weights` is given and not None, it must be a NumPy array of numeric
    weight values, or a (weights, validity) tuple, corresponding row-wise
    to any cube.dims.

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

    def __init__(self, arr, weights=None, ignore_missing=False, return_missing_as=NaN):
        arr, validity = as_separate_validity(arr)
        self.shape = arr.shape[1:] + arr.shape[1:]
        self.size = numpy.prod(self.shape, 0)

        if weights is not None:
            weights, weights_validity = as_separate_validity(weights)
            validity = (validity.T & weights_validity).T
            arr = (arr.T * weights).T

        self.arr = arr.astype(float).copy()
        self.arr[~validity] = NaN
        self.validity = validity
        if self.validity.ndim > 1:
            # if self.ignore_missing then we want to keep only complete cases,
            # like R's `use=na.or.complete`.
            self.validity = numpy.all(
                validity.T, axis=tuple(d for d in range(self.validity.ndim) if d != 1)
            )
        self.weights = weights
        self.ignore_missing = ignore_missing
        self.return_missing_as = return_missing_as
        if isinstance(self.return_missing_as, tuple):
            self.null = self.return_missing_as[0]
        else:
            self.null = self.return_missing_as

    def get_initial_regions(self, cube):
        """Return empty NumPy arrays to fill."""
        shape = cube.shape + self.shape
        if not shape:
            shape = (1,)
        corrcoefs = numpy.full(shape, self.null, dtype=float)
        return (corrcoefs,)

    def fill(self, coordinates, regions):
        """Fill the `regions` arrays with distributions contingent on coordinates.

        The given `regions` are assumed to be part of a (possibly larger) cube,
        one which has already been initialized (including any corner values).
        We will compute its common cells from the margins later in self.reduce.
        """
        (corrcoefs,) = self.flat_regions(regions)

        if self.ignore_missing:
            arr = self.arr[self.validity]
        else:
            arr = self.arr

        if coordinates is None:
            corrcoefs[:] = numpy.corrcoef(arr, rowvar=False)
        else:
            if self.ignore_missing:
                coordinates = coordinates[self.validity]
            for i, rowmask in self.bins(coordinates):
                corrcoefs[i] = numpy.corrcoef(arr[rowmask], rowvar=False)

    def reduce(self, cube, regions):
        """Return `regions` reduced to proper output."""
        (corrcoefs,) = regions
        if isinstance(self.return_missing_as, tuple):
            missings = numpy.isnan(corrcoefs)
            corrcoefs[missings] = self.null
            return corrcoefs, ~missings
        else:
            return corrcoefs


class xfunc_covariance(xfunc):
    """Estimate the covariance matrixes of an array contingent on a cube.

    The `arr` arg must be a NumPy array of numeric values to be analyzed,
    or a tuple of (values, validity) arrays, corresponding row-wise
    to any cube.dims.

    If `weights` is given and not None, it must be a NumPy array of numeric
    weight values, or a (weights, validity) tuple, corresponding row-wise
    to any cube.dims.

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

    def __init__(self, arr, weights=None, ignore_missing=False, return_missing_as=NaN):
        arr, validity = as_separate_validity(arr)
        self.shape = arr.shape[1:] + arr.shape[1:]
        self.size = numpy.prod(self.shape, 0)

        if weights is not None:
            weights, weights_validity = as_separate_validity(weights)
            validity = (validity.T & weights_validity).T
            weights = weights.copy()
            weights[~weights_validity] = NaN

        self.arr = arr.astype(float).copy()
        self.arr[~validity] = NaN
        self.validity = validity
        if self.validity.ndim > 1:
            # if self.ignore_missing then we want to keep only complete cases,
            # like R's `use=na.or.complete`.
            self.validity = numpy.all(
                validity.T, axis=tuple(d for d in range(self.validity.ndim) if d != 1)
            )
        self.weights = weights
        self.ignore_missing = ignore_missing
        self.return_missing_as = return_missing_as
        if isinstance(self.return_missing_as, tuple):
            self.null = self.return_missing_as[0]
        else:
            self.null = self.return_missing_as

    def get_initial_regions(self, cube):
        """Return empty NumPy arrays to fill."""
        shape = cube.shape + self.shape
        if not shape:
            shape = (1,)
        covs = numpy.full(shape, self.null, dtype=float)
        return (covs,)

    def fill(self, coordinates, regions):
        """Fill the `regions` arrays with distributions contingent on coordinates.

        The given `regions` are assumed to be part of a (possibly larger) cube,
        one which has already been initialized (including any corner values).
        We will compute its common cells from the margins later in self.reduce.
        """
        (covs,) = self.flat_regions(regions)

        if self.ignore_missing:
            arr = self.arr[self.validity]
        else:
            arr = self.arr

        if self.weights is None:
            aweights = None
        else:
            if self.ignore_missing:
                aweights = self.weights[self.validity]
            else:
                aweights = self.weights
            if len(aweights) == 0:
                aweights = None

        if coordinates is None:
            covs[:] = numpy.cov(arr.T, aweights=aweights)
        else:
            if self.ignore_missing:
                coordinates = coordinates[self.validity]
            for i, rowmask in self.bins(coordinates):
                if aweights is None:
                    w = None
                else:
                    w = aweights[rowmask]
                covs[i] = numpy.cov(arr[rowmask].T, aweights=w)

    def reduce(self, cube, regions):
        """Return `regions` reduced to proper output."""
        (covs,) = regions
        if isinstance(self.return_missing_as, tuple):
            missings = numpy.isnan(covs)
            covs[missings] = self.null
            return covs, ~missings
        else:
            return covs
