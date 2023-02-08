import numpy


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


class ffunc_count(ffunc):
    """A `count` frequency function.

    If `weights` are given and not None, it must be a NumPy array of numeric
    weight values, which correspond row-wise to any cube.dims.

    """

    def __init__(self, weights=None, N=None):
        self.weights = weights
        self.N = N

    def get_initial_regions(self, cube):
        """Return NumPy arrays to fill, empty except for corner values."""
        if self.weights is None:
            dtype = int
            if self.N is not None:
                corner_value = self.N
            elif cube.dims:
                corner_value = cube.dims[0].shape[0]
            elif self.weights is not None:
                corner_value = self.weights.shape[0]
            else:
                raise ValueError(
                    "Cannot determine counts with no N, dimensions, or weights."
                )
        else:
            dtype = float
            corner_value = self.weights.sum()

        counts = numpy.zeros(cube.working_shape, dtype=dtype)
        # Set the "grand total" value in the corner of each region.
        counts[cube.corner] = corner_value

        return (counts,)

    def fill(self, cube, regions):
        """Fill the `regions` arrays with distributions contingent on cube.dims.

        The given `regions` are assumed to be part of a (possibly larger) cube,
        one which has already been initialized (including any corner values).
        We will compute its common cells from the margins later in self.reduce.
        """
        (counts,) = regions
        if self.weights is None:

            def _fill(x_coords, x_rowids):
                counts[x_coords] = len(x_rowids)

        else:

            def _fill(x_coords, x_rowids):
                counts[x_coords] = self.weights[x_rowids].sum()

        cube.walk(_fill)

    def reduce(self, cube, regions):
        """Return `regions` with common cells calculated and margins removed."""
        # Calculate common slices by subtracting uncommon results from margins.
        (counts,) = regions
        cube._compute_common_cells_from_marginal_diffs(counts)

        # Cut the margins off our larger result.
        output = counts[cube.marginless]

        # Sometimes, marginal differencing can produce minutely different results
        # (within the machine's floating-point epsilon). For most values, this
        # has little effect, but when the value should be 0 but is just barely
        # not 0, it's easy to end up with e.g. 1e-09/1e-09=1 instead of 0/0=nan.
        if output.shape:
            output[numpy.isclose(output, 0)] = 0
        else:
            if numpy.isclose(output, 0):
                output = output.dtype(0)

        return (output,)


class ffunc_sum(ffunc):
    def __init__(self, summables, countables, weights=None):
        self.summables = numpy.asarray(summables)
        self.countables = numpy.asarray(countables)
        self.weights = weights
        if weights is not None:
            self.summables = (summables.T * weights).T
            self.countables = self.countables.copy()
            self.countables[weights == 0] = False

    def get_initial_regions(self, cube):
        """Return NumPy arrays to fill, empty except for corner values."""
        dtype = int if self.weights is None else float

        sums = numpy.zeros(cube.working_shape, dtype=self.summables.dtype)
        valid_counts = numpy.zeros(cube.working_shape, dtype=dtype)

        # Set the "grand total" value in the corner of each region.
        sums[cube.corner] = numpy.sum(self.summables, axis=0)
        valid_counts[cube.corner] = numpy.count_nonzero(self.countables, axis=0)

        return sums, valid_counts

    def fill(self, cube, regions):
        """Fill the `regions` arrays with distributions contingent on cube.dims.

        The given `regions` are assumed to be part of a (possibly larger) cube,
        one which has already been initialized (including any corner values).
        We will compute its common cells from the margins later in self.reduce.
        """
        sums, valid_counts = regions

        def _fill(x_coords, x_rowids):
            sums[x_coords] = numpy.sum(self.summables[x_rowids], axis=0)
            valid_counts[x_coords] = numpy.count_nonzero(
                self.countables[x_rowids], axis=0
            )

        cube.walk(_fill)

    def reduce(self, cube, regions):
        """Return `regions` with common cells calculated and margins removed."""
        # Calculate common slices by subtracting uncommon results from margins.
        sums, valid_counts = regions
        cube._compute_common_cells_from_marginal_diffs(sums)
        cube._compute_common_cells_from_marginal_diffs(valid_counts)

        # Cut the margins off our larger result.
        sums = sums[cube.marginless]
        valid_counts = valid_counts[cube.marginless]

        # Sometimes, marginal differencing can produce minutely different results
        # (within the machine's floating-point epsilon). For most values, this
        # has little effect, but when the value should be 0 but is just barely
        # not 0, it's easy to end up with e.g. 1e-09/1e-09=1 instead of 0/0=nan.
        no_valids_mask = numpy.isclose(valid_counts, 0)
        sums[no_valids_mask] = 0
        valid_counts[no_valids_mask] = 0

        return sums, valid_counts
