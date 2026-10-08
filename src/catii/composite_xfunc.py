"""Composite xfuncs that share intermediate computations.

This module provides:
1. `group_xfuncs`: Groups compatible xfuncs that can share computations
2. `CompositeXfunc`: An xfunc that wraps multiple compatible xfuncs and
   computes their shared intermediates once

When xfuncs like mean and stddev operate on the same data, they compute
many of the same intermediate values (weighted sums, weighted counts, means).
CompositeXfunc computes these once and shares them.

DEPENDENCY GRAPH:
-----------------
                    ┌─────────────┐
                    │   coords    │
                    └──────┬──────┘
                           │
           ┌───────────────┼───────────────┐
           │               │               │
           ▼               ▼               ▼
    ┌──────────┐    ┌──────────┐    ┌──────────────┐
    │    N     │    │valid_cnt │    │  value_sums  │
    │(unwt cnt)│    │ (wt cnt) │    │   (wt sum)   │
    └────┬─────┘    └────┬─────┘    └──────┬───────┘
         │               │                 │
         │               └────────┬────────┘
         │                        │
         │                        ▼
         │               ┌──────────────┐
         │               │    means     │
         │               └──────┬───────┘
         │                      │
         └──────────┬───────────┘
                    │
                    ▼
           ┌──────────────┐
           │variance_sums │
           └──────┬───────┘
                  │
                  ▼
           ┌──────────────┐
           │   stddev     │
           └──────────────┘

XFUNC REQUIREMENTS:
-------------------
Each xfunc declares what intermediates it needs. Dependencies are implicit
in the computation order.

xfunc_valid_count: N, valid_counts, missing_counts
xfunc_sum:         N, valid_counts, missing_counts, value_sums
xfunc_mean:        N, valid_counts, missing_counts, value_sums, means
xfunc_stddev:      N, valid_counts, missing_counts, value_sums, weight_sums, means, variance_sums
"""

import numpy


# =============================================================================
# What intermediates each xfunc type needs
# =============================================================================

XFUNC_REQUIREMENTS = {
    "xfunc_valid_count": ["N", "valid_counts", "missing_counts"],
    "xfunc_sum": ["N", "valid_counts", "missing_counts", "value_sums"],
    "xfunc_mean": ["N", "valid_counts", "missing_counts", "value_sums"],
    "xfunc_stddev": [
        "N",
        "valid_counts",
        "missing_counts",
        "value_sums",
        "weight_sums",
        "means",
        "variance_sums",
    ],
}


# =============================================================================
# Strategy classes for filling xfunc regions
# =============================================================================


class XfuncFillStrategy:
    """Base class for xfunc fill strategies.

    Each strategy knows how to fill regions for a specific xfunc type
    from a cache of precomputed intermediates.
    """

    def fill_regions(self, xfunc, regions, cache, weights, col=None):
        """Fill xfunc regions from cache.

        Args:
            xfunc: The xfunc instance being filled.
            regions: Tuple of region arrays to fill.
            cache: Dict of precomputed intermediate values.
            weights: Weights array or None.
            col: Column index for 2D case, or None for 1D case.
        """
        raise NotImplementedError

    def _set_region(self, arr, value, col):
        """Set region array values, handling 1D vs 2D indexing."""
        if col is None:
            arr[:] = value
        else:
            arr[:, col] = value


class MeanFillStrategy(XfuncFillStrategy):
    """Strategy for filling mean xfunc regions."""

    def fill_regions(self, xfunc, regions, cache, weights, col=None):
        if xfunc.ignore_missing:
            sums, valid_counts = regions
        else:
            sums, valid_counts, missing_counts = regions

        self._set_region(sums, cache["value_sums"], col)
        self._set_region(valid_counts, cache["valid_counts"], col)
        if not xfunc.ignore_missing:
            self._set_region(missing_counts, cache.get("missing_counts", 0), col)


class StddevFillStrategy(XfuncFillStrategy):
    """Strategy for filling stddev xfunc regions."""

    def fill_regions(self, xfunc, regions, cache, weights, col=None):
        if xfunc.ignore_missing:
            stddevs, valid_counts = regions
        else:
            stddevs, valid_counts, missing_counts = regions

        N = cache["N"]
        varsums = cache.get("variance_sums")
        weight_sums = cache.get("weight_sums")

        if varsums is not None:
            with numpy.errstate(divide="ignore", invalid="ignore"):
                if weights is None:
                    stddev_values = numpy.sqrt(varsums / (N - 1))
                else:
                    stddev_values = numpy.sqrt((varsums / weight_sums) * (N / (N - 1)))
                self._set_region(stddevs, stddev_values, col)

        self._set_region(valid_counts, N, col)
        if not xfunc.ignore_missing:
            self._set_region(missing_counts, cache.get("missing_counts", 0), col)


class SumFillStrategy(XfuncFillStrategy):
    """Strategy for filling sum xfunc regions."""

    def fill_regions(self, xfunc, regions, cache, weights, col=None):
        if xfunc.ignore_missing:
            sums, valid_counts = regions
        else:
            sums, valid_counts, missing_counts = regions

        self._set_region(sums, cache["value_sums"], col)
        self._set_region(valid_counts, cache["N"], col)
        if not xfunc.ignore_missing:
            self._set_region(missing_counts, cache.get("missing_counts", 0), col)


class ValidCountFillStrategy(XfuncFillStrategy):
    """Strategy for filling valid_count xfunc regions."""

    def fill_regions(self, xfunc, regions, cache, weights, col=None):
        if xfunc.ignore_missing:
            counts, valid_counts = regions
        else:
            counts, valid_counts, missing_counts = regions

        self._set_region(counts, cache["valid_counts"], col)
        self._set_region(valid_counts, cache["N"], col)
        if not xfunc.ignore_missing:
            self._set_region(missing_counts, cache.get("missing_counts", 0), col)


# Registry mapping xfunc type names to their fill strategies
XFUNC_FILL_STRATEGIES = {
    "xfunc_mean": MeanFillStrategy(),
    "xfunc_stddev": StddevFillStrategy(),
    "xfunc_sum": SumFillStrategy(),
    "xfunc_valid_count": ValidCountFillStrategy(),
}


# =============================================================================
# Grouping logic
# =============================================================================

# Types that can share intermediate computations
COMPATIBLE_XFUNC_TYPES = {"xfunc_mean", "xfunc_stddev", "xfunc_valid_count", "xfunc_sum"}


def can_share_intermediates(xfunc1, xfunc2):
    """Check if two xfuncs can share intermediate computations.

    They can share if they:
    1. Are both compatible types
    2. Have the same weights array (identity, since xfuncs don't copy weights)
    3. Have the same ignore_missing setting
    4. Have equal validity arrays
    5. Have equal summables arrays
    """
    type1 = type(xfunc1).__name__
    type2 = type(xfunc2).__name__

    if type1 not in COMPATIBLE_XFUNC_TYPES or type2 not in COMPATIBLE_XFUNC_TYPES:
        return False

    # Weights are not copied by xfuncs, so identity check is appropriate
    if getattr(xfunc1, "weights", None) is not getattr(xfunc2, "weights", None):
        return False

    if getattr(xfunc1, "ignore_missing", False) != getattr(xfunc2, "ignore_missing", False):
        return False

    # Validity arrays are derived from factvar, need value comparison
    v1 = getattr(xfunc1, "validity", None)
    v2 = getattr(xfunc2, "validity", None)
    if v1 is None or v2 is None:
        return False
    if v1.shape != v2.shape or not numpy.array_equal(v1, v2):
        return False

    # Countables must both exist and match, or both not exist
    c1 = getattr(xfunc1, "countables", None)
    c2 = getattr(xfunc2, "countables", None)
    if (c1 is None) != (c2 is None):
        return False
    if c1 is not None and c2 is not None:
        if c1.shape != c2.shape or not numpy.allclose(c1, c2, equal_nan=True):
            return False

    # Summables are derived from factvar, need value comparison
    def get_summables(xf):
        ws = getattr(xf, "wsummables", None)
        if ws is not None:
            return ws
        return getattr(xf, "summables", None)

    s1 = get_summables(xfunc1)
    s2 = get_summables(xfunc2)

    if s1 is None and s2 is None:
        return True
    if s1 is None or s2 is None:
        return True  # One has values, one doesn't (e.g., valid_count vs mean)
    if s1.shape != s2.shape:
        return False

    return numpy.allclose(s1, s2, equal_nan=True)


def group_xfuncs(xfuncs):
    """Group xfuncs that can share intermediate computations.

    Returns a list of groups, where xfuncs in the same group can share
    intermediates.
    """
    groups = []

    for xfunc in xfuncs:
        found_group = False
        for group in groups:
            if group and can_share_intermediates(group[0], xfunc):
                group.append(xfunc)
                found_group = True
                break

        if not found_group:
            groups.append([xfunc])

    return groups


# =============================================================================
# CompositeXfunc
# =============================================================================


class CompositeXfunc:
    """An xfunc that wraps multiple compatible xfuncs and shares intermediates.

    This class:
    1. Takes a list of compatible xfuncs (same factvar, weights, ignore_missing)
    2. Determines what intermediates are needed by all of them
    3. Computes each intermediate once during fill()
    4. Returns results for all wrapped xfuncs from reduce()

    Usage:
        fmean = xfunc_mean(factvar)
        fstddev = xfunc_stddev(factvar)
        composite = CompositeXfunc([fmean, fstddev])

        # Use like any other xfunc:
        regions = composite.get_initial_regions(cube)
        composite.fill(coordinates, regions)
        means, stddevs = composite.reduce(cube, regions)
    """

    def __init__(self, xfuncs):
        """Initialize with a list of compatible xfuncs.

        Args:
            xfuncs: List of xfunc instances that can share intermediates.
                    They must all operate on the same factvar with same weights.
        """
        if not xfuncs:
            raise ValueError("CompositeXfunc requires at least one xfunc")

        self.xfuncs = xfuncs
        self.reference = xfuncs[0]  # Use first xfunc as reference for data arrays

        # Determine what intermediates we need
        self.required_intermediates = self._collect_requirements()

        # Get data arrays from reference xfunc
        self.weights = getattr(self.reference, "weights", None)
        self.validity = getattr(self.reference, "validity", None)
        self.countables = getattr(self.reference, "countables", None)
        self.ignore_missing = getattr(self.reference, "ignore_missing", False)

        # Get summables (wsummables for weighted, summables for unweighted)
        if hasattr(self.reference, "wsummables"):
            self.wsummables = self.reference.wsummables
        elif hasattr(self.reference, "summables"):
            self.wsummables = self.reference.summables
        else:
            self.wsummables = None

        # For stddev, we also need unweighted summables
        self.summables = getattr(self.reference, "summables", None)

        # Determine dimensionality
        if self.wsummables is not None:
            self.ndim = self.wsummables.ndim
            self.num_cols = self.wsummables.shape[1] if self.ndim == 2 else 1
        elif self.countables is not None:
            self.ndim = self.countables.ndim
            self.num_cols = self.countables.shape[1] if self.ndim == 2 else 1
        else:
            self.ndim = 1
            self.num_cols = 1

    def _collect_requirements(self):
        """Collect all intermediate requirements from wrapped xfuncs."""
        required = []
        for xfunc in self.xfuncs:
            xfunc_name = type(xfunc).__name__
            reqs = XFUNC_REQUIREMENTS.get(xfunc_name, [])
            for req in reqs:
                if req not in required:
                    required.append(req)
        return required

    def get_initial_regions(self, cube):
        """Return initial regions for all wrapped xfuncs.

        Returns a list of regions, one per wrapped xfunc.
        """
        all_regions = []
        for xfunc in self.xfuncs:
            regions = xfunc.get_initial_regions(cube)
            all_regions.append(regions)
        return all_regions

    def fill(self, coordinates, all_regions):
        """Fill regions for all wrapped xfuncs using shared intermediates.

        Args:
            coordinates: Bin coordinates for each row, or None for no-coordinates case
            all_regions: List of regions, one per wrapped xfunc
        """
        if coordinates is None:
            self._fill_no_coordinates(all_regions)
        else:
            self._fill_with_coordinates(coordinates, all_regions)

    def _fill_no_coordinates(self, all_regions):
        """Fill when there are no coordinates (single-bin case)."""
        # For the no-coordinates case, we sum over all rows
        cache = {}

        # Compute required intermediates
        if "N" in self.required_intermediates:
            cache["N"] = numpy.count_nonzero(self.validity, axis=0)

        if "valid_counts" in self.required_intermediates:
            cache["valid_counts"] = numpy.sum(self.countables, axis=0)

        if "missing_counts" in self.required_intermediates:
            cache["missing_counts"] = numpy.sum(~self.validity, axis=0)

        if "value_sums" in self.required_intermediates and self.wsummables is not None:
            cache["value_sums"] = numpy.nansum(self.wsummables, axis=0)

        if "weight_sums" in self.required_intermediates and self.weights is not None:
            cache["weight_sums"] = numpy.sum(self.weights, axis=0)

        if "means" in self.required_intermediates and "value_sums" in cache:
            with numpy.errstate(divide="ignore", invalid="ignore"):
                cache["means"] = cache["value_sums"] / cache["valid_counts"]

        if "variance_sums" in self.required_intermediates and self.summables is not None:
            means = cache.get("means")
            if means is not None:
                squared_variances = (self.summables - means) ** 2
                if self.weights is not None:
                    squared_variances = (squared_variances.T * self.weights).T
                cache["variance_sums"] = numpy.nansum(squared_variances, axis=0)

        # Fill each xfunc's regions from cache
        for xfunc, regions in zip(self.xfuncs, all_regions):
            self._fill_xfunc_regions(xfunc, regions, cache)

    def _fill_with_coordinates(self, coordinates, all_regions):
        """Fill when we have coordinates (multi-bin case)."""
        # Get the first xfunc's regions to determine size
        first_regions = all_regions[0]
        flat_regions = self.xfuncs[0].flat_regions(first_regions)
        size = flat_regions[0].shape[0]

        if self.ndim == 1:
            cache = self._compute_intermediates_1d(coordinates, size)
            for xfunc, regions in zip(self.xfuncs, all_regions):
                flat_regs = xfunc.flat_regions(regions)
                self._fill_xfunc_regions(xfunc, flat_regs, cache)
        else:
            # For 2D, compute per column
            for col in range(self.num_cols):
                cache = self._compute_intermediates_1d_column(coordinates, size, col)
                for xfunc, regions in zip(self.xfuncs, all_regions):
                    flat_regs = xfunc.flat_regions(regions)
                    self._fill_xfunc_regions(xfunc, flat_regs, cache, col)

    def _compute_intermediates_1d(self, coordinates, size):
        """Compute all required intermediates for 1D case."""
        cache = {}

        if "N" in self.required_intermediates:
            cache["N"] = numpy.bincount(coordinates, minlength=size)[:size]

        if "valid_counts" in self.required_intermediates:
            cache["valid_counts"] = numpy.bincount(
                coordinates, weights=self.countables, minlength=size
            )[:size]

        if "missing_counts" in self.required_intermediates:
            cache["missing_counts"] = numpy.bincount(
                coordinates, weights=~self.validity, minlength=size
            )[:size].astype(int)

        if "value_sums" in self.required_intermediates and self.wsummables is not None:
            cache["value_sums"] = numpy.bincount(
                coordinates, weights=self.wsummables, minlength=size
            )[:size]

        if "weight_sums" in self.required_intermediates and self.weights is not None:
            cache["weight_sums"] = numpy.bincount(
                coordinates, weights=self.weights, minlength=size
            )[:size]

        if "means" in self.required_intermediates and "value_sums" in cache:
            with numpy.errstate(divide="ignore", invalid="ignore"):
                cache["means"] = cache["value_sums"] / cache["valid_counts"]

        if "variance_sums" in self.required_intermediates and self.summables is not None:
            means = cache.get("means")
            if means is not None:
                squared_variances = (self.summables - means[coordinates]) ** 2
                if self.weights is not None:
                    squared_variances = squared_variances * self.weights
                cache["variance_sums"] = numpy.bincount(
                    coordinates, weights=squared_variances, minlength=size
                )[:size]

        return cache

    def _compute_intermediates_1d_column(self, coordinates, size, col):
        """Compute all required intermediates for one column of 2D data."""
        cache = {}

        if "N" in self.required_intermediates:
            # N is the same for all columns
            cache["N"] = numpy.bincount(coordinates, minlength=size)[:size]

        if "valid_counts" in self.required_intermediates:
            cache["valid_counts"] = numpy.bincount(
                coordinates, weights=self.countables[:, col], minlength=size
            )[:size]

        if "missing_counts" in self.required_intermediates:
            cache["missing_counts"] = numpy.bincount(
                coordinates, weights=~self.validity[:, col], minlength=size
            )[:size].astype(int)

        if "value_sums" in self.required_intermediates and self.wsummables is not None:
            cache["value_sums"] = numpy.bincount(
                coordinates, weights=self.wsummables[:, col], minlength=size
            )[:size]

        if "weight_sums" in self.required_intermediates and self.weights is not None:
            cache["weight_sums"] = numpy.bincount(
                coordinates, weights=self.weights, minlength=size
            )[:size]

        if "means" in self.required_intermediates and "value_sums" in cache:
            with numpy.errstate(divide="ignore", invalid="ignore"):
                cache["means"] = cache["value_sums"] / cache["valid_counts"]

        if "variance_sums" in self.required_intermediates and self.summables is not None:
            means = cache.get("means")
            if means is not None:
                squared_variances = (self.summables[:, col] - means[coordinates]) ** 2
                if self.weights is not None:
                    squared_variances = squared_variances * self.weights
                cache["variance_sums"] = numpy.bincount(
                    coordinates, weights=squared_variances, minlength=size
                )[:size]

        return cache

    def _fill_xfunc_regions(self, xfunc, regions, cache, col=None):
        """Fill one xfunc's regions from the cache.
        
        Args:
            xfunc: The xfunc instance being filled.
            regions: Tuple of region arrays to fill.
            cache: Dict of precomputed intermediate values.
            col: Column index for 2D case, or None for 1D case.
        """
        xfunc_name = type(xfunc).__name__
        strategy = XFUNC_FILL_STRATEGIES.get(xfunc_name)
        if strategy:
            strategy.fill_regions(xfunc, regions, cache, self.weights, col)

    def reduce(self, cube, all_regions):
        """Reduce regions for all wrapped xfuncs.

        Returns a tuple of results, one per wrapped xfunc.
        """
        results = []
        for xfunc, regions in zip(self.xfuncs, all_regions):
            result = xfunc.reduce(cube, regions)
            results.append(result)
        return tuple(results)
