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
            validity = validity & weights_validity
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
            validity = validity & weights_validity
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
        else:
            sums, valid_counts, missing_counts = regions

        sums = self.adjust_zeros(sums, self.null, condition=valid_counts == 0)

        with numpy.errstate(divide="ignore", invalid="ignore"):
            means = sums / valid_counts

        if not self.ignore_missing:
            if means.shape:
                means[missing_counts.nonzero()] = self.null
            elif missing_counts != 0:
                means = means.dtype.type(self.null)

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
            summables = summables.copy()
            countables = validity.astype(int)
        else:
            weights, weights_validity = as_separate_validity(weights)
            validity = (validity.T & weights_validity).T
            summables = (summables.T * weights).T
            countables = (validity.T * weights).T

        summables[~validity] = float("nan")
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

        if self.ignore_missing:
            stddevs, valid_counts = regions
        else:
            stddevs, missing_counts = regions

        # This can be called thousands of times, so it's critical
        # to perform as few passes over the data as possible.
        # We set summables/countables[~validity] = 0 so there's
        # no need to filter them out again here.

        if coordinates is None:
            sums = numpy.nansum(self.summables, axis=0)
            wcounts = numpy.nansum(self.countables, axis=0)
            means = sums / wcounts
            squared_variances = (self.summables - means) ** 2
            if self.weights is not None:
                squared_variances = (squared_variances.T * self.weights).T

            varsums = numpy.nansum(squared_variances, axis=0)
            N = numpy.count_nonzero(self.validity, axis=0)

            with numpy.errstate(divide="ignore", invalid="ignore"):
                if self.weights is None:
                    stddevs[:] = numpy.sqrt(varsums / (N - 1))
                else:
                    stddevs[:] = numpy.sqrt((varsums / wcounts) * (N / (N - 1)))

            if self.ignore_missing:
                valid_counts[:] = N
            else:
                missing_counts[:] = numpy.sum(~self.validity, axis=0)
        else:
            size = stddevs.shape[0]
            if self.summables.ndim == 1:
                if self.ignore_missing:
                    coords = coordinates[self.validity]
                    summables = self.summables[self.validity]
                    countables = self.countables[self.validity]
                else:
                    coords = coordinates
                    summables = self.summables
                    countables = self.countables

                sums = numpy.bincount(coords, weights=summables, minlength=size)
                wcounts = numpy.bincount(coords, weights=countables, minlength=size)
                means = sums / wcounts

                # Neat trick: means[coords] gives us the appropriate
                # classified mean for each input row.
                squared_variances = (summables - means[coords]) ** 2
                if self.weights is not None:
                    squared_variances *= self.weights
                varsums = numpy.bincount(
                    coords, weights=squared_variances, minlength=size
                )
                N = numpy.bincount(
                    coords,
                    weights=None if self.ignore_missing else self.validity,
                    minlength=size,
                )

                with numpy.errstate(divide="ignore", invalid="ignore"):
                    if self.weights is None:
                        stddevs[:] = numpy.sqrt(varsums / (N - 1))
                    else:
                        stddevs[:] = numpy.sqrt((varsums / wcounts) * (N / (N - 1)))
            elif self.summables.ndim == 2:
                for i in range(self.summables.shape[1]):
                    sums = numpy.bincount(
                        coordinates, weights=self.summables[:, i], minlength=size
                    )
                    wcounts = numpy.bincount(
                        coordinates, weights=self.countables[:, i], minlength=size
                    )
                    means = sums / wcounts

                    squared_variances = (self.summables[:, i] - means[coordinates]) ** 2
                    if self.weights is not None:
                        squared_variances *= self.weights
                    varsums = numpy.bincount(
                        coordinates, weights=squared_variances, minlength=size
                    )
                    N = numpy.bincount(
                        coordinates, weights=self.validity[:, i], minlength=size
                    )

                    with numpy.errstate(divide="ignore", invalid="ignore"):
                        if self.weights is None:
                            stddevs[:, i] = numpy.sqrt(varsums / (N - 1))
                        else:
                            stddevs[:, i] = numpy.sqrt(
                                (varsums / wcounts) * (N / (N - 1))
                            )

            if self.ignore_missing:
                valid_counts[:] = N
            else:
                missing_counts[:] = numpy.bincount(
                    coordinates, weights=~self.validity, minlength=size
                )

    def reduce(self, cube, regions):
        """Return `regions` reduced to proper output."""
        if self.ignore_missing:
            stddevs, valid_counts = regions
            stddevs = self.adjust_zeros(stddevs, self.null, condition=valid_counts == 0)
        else:
            stddevs, missing_counts = regions
            stddevs = self.adjust_zeros(
                stddevs, self.null, condition=missing_counts != 0
            )

        if isinstance(self.return_missing_as, tuple):
            if self.ignore_missing:
                validity = valid_counts != 0
            else:
                validity = missing_counts == 0
            return stddevs, validity
        else:
            return stddevs


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

        dtype = float if numpy.isnan(self.null) else self.values.dtype
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
