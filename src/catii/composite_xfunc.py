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
# Intermediate computation functions
# =============================================================================

def compute_N(coords, size):
    """Compute unweighted count per bin."""
    return numpy.bincount(coords, minlength=size)[:size]


def compute_valid_counts(coords, size, countables):
    """Compute weighted valid counts per bin."""
    return numpy.bincount(coords, weights=countables, minlength=size)[:size]


def compute_missing_counts(coords, size, validity):
    """Compute count of missing values per bin."""
    return numpy.bincount(coords, weights=~validity, minlength=size)[:size].astype(int)


def compute_value_sums(coords, size, wsummables):
    """Compute weighted value sums per bin."""
    return numpy.bincount(coords, weights=wsummables, minlength=size)[:size]


def compute_weight_sums(coords, size, weights):
    """Compute sum of weights per bin."""
    return numpy.bincount(coords, weights=weights, minlength=size)[:size]


def compute_means(value_sums, valid_counts):
    """Compute means from sums and counts."""
    with numpy.errstate(divide="ignore", invalid="ignore"):
        return value_sums / valid_counts


def compute_variance_sums(coords, size, summables, means, weights=None):
    """Compute sum of squared deviations per bin."""
    squared_variances = (summables - means[coords]) ** 2
    if weights is not None:
        squared_variances = squared_variances * weights
    return numpy.bincount(coords, weights=squared_variances, minlength=size)[:size]


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
# Grouping logic
# =============================================================================


def can_share_intermediates(xfunc1, xfunc2):
    """Check if two xfuncs can share intermediate computations.

    They can share if they operate on the same fact variable with the
    same weights and ignore_missing setting.
    """
    compatible_types = {"xfunc_mean", "xfunc_stddev", "xfunc_valid_count", "xfunc_sum"}
    type1 = type(xfunc1).__name__
    type2 = type(xfunc2).__name__

    if type1 not in compatible_types or type2 not in compatible_types:
        return False

    # Check weights identity (weights is NOT copied by xfuncs)
    w1 = getattr(xfunc1, "weights", None)
    w2 = getattr(xfunc2, "weights", None)
    if (w1 is None) != (w2 is None):
        return False
    if w1 is not None and w1 is not w2:
        return False

    # Check ignore_missing
    im1 = getattr(xfunc1, "ignore_missing", False)
    im2 = getattr(xfunc2, "ignore_missing", False)
    if im1 != im2:
        return False

    # Check validity shape and values
    v1 = getattr(xfunc1, "validity", None)
    v2 = getattr(xfunc2, "validity", None)
    if v1 is None or v2 is None:
        return False
    if getattr(v1, "shape", None) != getattr(v2, "shape", None):
        return False
    if not numpy.array_equal(v1, v2):
        return False

    # Check countables are equal
    c1 = getattr(xfunc1, "countables", None)
    c2 = getattr(xfunc2, "countables", None)
    if c1 is None or c2 is None:
        return False
    if getattr(c1, "shape", None) != getattr(c2, "shape", None):
        return False
    if not numpy.allclose(c1, c2, equal_nan=True):
        return False

    # Check that the actual values (wsummables/summables) are the same
    # This catches the case where countables match but values differ
    def get_wsummables(xf):
        if hasattr(xf, "wsummables"):
            return xf.wsummables
        elif hasattr(xf, "summables"):
            return xf.summables
        return None

    ws1 = get_wsummables(xfunc1)
    ws2 = get_wsummables(xfunc2)

    # If both have wsummables, they must be equal on valid positions
    if ws1 is not None and ws2 is not None:
        if ws1.shape != ws2.shape:
            return False
        # Compare only valid positions (where neither is NaN)
        valid_mask = ~(numpy.isnan(ws1) | numpy.isnan(ws2))
        if valid_mask.any():
            if not numpy.allclose(ws1[valid_mask], ws2[valid_mask]):
                return False
        # Also check that NaN positions match
        if not numpy.array_equal(numpy.isnan(ws1), numpy.isnan(ws2)):
            return False
    elif (ws1 is None) != (ws2 is None):
        # One has values, one doesn't: can still share countables
        # but this is an edge case (e.g., valid_count vs mean)
        pass

    return True


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
            cache["means"] = compute_means(cache["value_sums"], cache["valid_counts"])

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
                    self._fill_xfunc_regions_column(xfunc, flat_regs, cache, col)

    def _compute_intermediates_1d(self, coordinates, size):
        """Compute all required intermediates for 1D case."""
        cache = {}

        if "N" in self.required_intermediates:
            cache["N"] = compute_N(coordinates, size)

        if "valid_counts" in self.required_intermediates:
            cache["valid_counts"] = compute_valid_counts(
                coordinates, size, self.countables
            )

        if "missing_counts" in self.required_intermediates:
            cache["missing_counts"] = compute_missing_counts(
                coordinates, size, self.validity
            )

        if "value_sums" in self.required_intermediates and self.wsummables is not None:
            cache["value_sums"] = compute_value_sums(
                coordinates, size, self.wsummables
            )

        if "weight_sums" in self.required_intermediates and self.weights is not None:
            cache["weight_sums"] = compute_weight_sums(coordinates, size, self.weights)

        if "means" in self.required_intermediates and "value_sums" in cache:
            cache["means"] = compute_means(cache["value_sums"], cache["valid_counts"])

        if "variance_sums" in self.required_intermediates and self.summables is not None:
            means = cache.get("means")
            if means is not None:
                cache["variance_sums"] = compute_variance_sums(
                    coordinates, size, self.summables, means, self.weights
                )

        return cache

    def _compute_intermediates_1d_column(self, coordinates, size, col):
        """Compute all required intermediates for one column of 2D data."""
        cache = {}

        if "N" in self.required_intermediates:
            # N is the same for all columns
            cache["N"] = compute_N(coordinates, size)

        if "valid_counts" in self.required_intermediates:
            cache["valid_counts"] = compute_valid_counts(
                coordinates, size, self.countables[:, col]
            )

        if "missing_counts" in self.required_intermediates:
            cache["missing_counts"] = compute_missing_counts(
                coordinates, size, self.validity[:, col]
            )

        if "value_sums" in self.required_intermediates and self.wsummables is not None:
            cache["value_sums"] = compute_value_sums(
                coordinates, size, self.wsummables[:, col]
            )

        if "weight_sums" in self.required_intermediates and self.weights is not None:
            cache["weight_sums"] = compute_weight_sums(coordinates, size, self.weights)

        if "means" in self.required_intermediates and "value_sums" in cache:
            cache["means"] = compute_means(cache["value_sums"], cache["valid_counts"])

        if "variance_sums" in self.required_intermediates and self.summables is not None:
            means = cache.get("means")
            if means is not None:
                cache["variance_sums"] = compute_variance_sums(
                    coordinates, size, self.summables[:, col], means, self.weights
                )

        return cache

    def _fill_xfunc_regions(self, xfunc, regions, cache):
        """Fill one xfunc's regions from the cache."""
        xfunc_name = type(xfunc).__name__

        if xfunc_name == "xfunc_mean":
            self._fill_mean_regions(xfunc, regions, cache)
        elif xfunc_name == "xfunc_stddev":
            self._fill_stddev_regions(xfunc, regions, cache)
        elif xfunc_name == "xfunc_sum":
            self._fill_sum_regions(xfunc, regions, cache)
        elif xfunc_name == "xfunc_valid_count":
            self._fill_valid_count_regions(xfunc, regions, cache)

    def _fill_xfunc_regions_column(self, xfunc, regions, cache, col):
        """Fill one column of one xfunc's regions from the cache."""
        xfunc_name = type(xfunc).__name__

        if xfunc_name == "xfunc_mean":
            self._fill_mean_regions_column(xfunc, regions, cache, col)
        elif xfunc_name == "xfunc_stddev":
            self._fill_stddev_regions_column(xfunc, regions, cache, col)
        elif xfunc_name == "xfunc_sum":
            self._fill_sum_regions_column(xfunc, regions, cache, col)
        elif xfunc_name == "xfunc_valid_count":
            self._fill_valid_count_regions_column(xfunc, regions, cache, col)

    def _fill_mean_regions(self, xfunc, regions, cache):
        """Fill mean xfunc regions."""
        if xfunc.ignore_missing:
            sums, valid_counts = regions
        else:
            sums, valid_counts, missing_counts = regions

        sums[:] = cache["value_sums"]
        valid_counts[:] = cache["valid_counts"]
        if not xfunc.ignore_missing:
            missing_counts[:] = cache.get("missing_counts", 0)

    def _fill_mean_regions_column(self, xfunc, regions, cache, col):
        """Fill mean xfunc regions for one column."""
        if xfunc.ignore_missing:
            sums, valid_counts = regions
        else:
            sums, valid_counts, missing_counts = regions

        sums[:, col] = cache["value_sums"]
        valid_counts[:, col] = cache["valid_counts"]
        if not xfunc.ignore_missing:
            missing_counts[:, col] = cache.get("missing_counts", 0)

    def _fill_stddev_regions(self, xfunc, regions, cache):
        """Fill stddev xfunc regions."""
        if xfunc.ignore_missing:
            stddevs, valid_counts = regions
        else:
            stddevs, valid_counts, missing_counts = regions

        N = cache["N"]
        varsums = cache.get("variance_sums")
        weight_sums = cache.get("weight_sums")

        if varsums is not None:
            with numpy.errstate(divide="ignore", invalid="ignore"):
                if self.weights is None:
                    stddevs[:] = numpy.sqrt(varsums / (N - 1))
                else:
                    stddevs[:] = numpy.sqrt((varsums / weight_sums) * (N / (N - 1)))

        valid_counts[:] = N
        if not xfunc.ignore_missing:
            missing_counts[:] = cache.get("missing_counts", 0)

    def _fill_stddev_regions_column(self, xfunc, regions, cache, col):
        """Fill stddev xfunc regions for one column."""
        if xfunc.ignore_missing:
            stddevs, valid_counts = regions
        else:
            stddevs, valid_counts, missing_counts = regions

        N = cache["N"]
        varsums = cache.get("variance_sums")
        weight_sums = cache.get("weight_sums")

        if varsums is not None:
            with numpy.errstate(divide="ignore", invalid="ignore"):
                if self.weights is None:
                    stddevs[:, col] = numpy.sqrt(varsums / (N - 1))
                else:
                    stddevs[:, col] = numpy.sqrt((varsums / weight_sums) * (N / (N - 1)))

        valid_counts[:, col] = N
        if not xfunc.ignore_missing:
            missing_counts[:, col] = cache.get("missing_counts", 0)

    def _fill_sum_regions(self, xfunc, regions, cache):
        """Fill sum xfunc regions."""
        if xfunc.ignore_missing:
            sums, valid_counts = regions
        else:
            sums, valid_counts, missing_counts = regions

        sums[:] = cache["value_sums"]
        valid_counts[:] = cache["N"]
        if not xfunc.ignore_missing:
            missing_counts[:] = cache.get("missing_counts", 0)

    def _fill_sum_regions_column(self, xfunc, regions, cache, col):
        """Fill sum xfunc regions for one column."""
        if xfunc.ignore_missing:
            sums, valid_counts = regions
        else:
            sums, valid_counts, missing_counts = regions

        sums[:, col] = cache["value_sums"]
        valid_counts[:, col] = cache["N"]
        if not xfunc.ignore_missing:
            missing_counts[:, col] = cache.get("missing_counts", 0)

    def _fill_valid_count_regions(self, xfunc, regions, cache):
        """Fill valid_count xfunc regions."""
        if xfunc.ignore_missing:
            counts, valid_counts = regions
        else:
            counts, valid_counts, missing_counts = regions

        counts[:] = cache["valid_counts"]
        valid_counts[:] = cache["N"]
        if not xfunc.ignore_missing:
            missing_counts[:] = cache.get("missing_counts", 0)

    def _fill_valid_count_regions_column(self, xfunc, regions, cache, col):
        """Fill valid_count xfunc regions for one column."""
        if xfunc.ignore_missing:
            counts, valid_counts = regions
        else:
            counts, valid_counts, missing_counts = regions

        counts[:, col] = cache["valid_counts"]
        valid_counts[:, col] = cache["N"]
        if not xfunc.ignore_missing:
            missing_counts[:, col] = cache.get("missing_counts", 0)

    def reduce(self, cube, all_regions):
        """Reduce regions for all wrapped xfuncs.

        Returns a tuple of results, one per wrapped xfunc.
        """
        results = []
        for xfunc, regions in zip(self.xfuncs, all_regions):
            result = xfunc.reduce(cube, regions)
            results.append(result)
        return tuple(results)
