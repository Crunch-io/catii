"""Tests for composite xfuncs that share intermediate computations."""

import numpy

from catii import xcube
from catii.composite_xfunc import CompositeXfunc, group_xfuncs
from catii.xfuncs import xfunc_mean, xfunc_stddev

from .. import arr_eq


class TestCompositeXfuncGrouping:
    """Test that xfuncs are correctly grouped by compatibility."""

    def test_group_same_factvar_no_weights(self):
        """xfuncs with same factvar and no weights should be grouped together."""
        factvar = numpy.arange(8, dtype=float)
        fmean = xfunc_mean(factvar)
        fstddev = xfunc_stddev(factvar)

        groups = group_xfuncs([fmean, fstddev])

        assert len(groups) == 1
        assert len(groups[0]) == 2

    def test_group_different_factvar(self):
        """xfuncs with different factvars should be in separate groups."""
        factvar1 = numpy.arange(8, dtype=float)
        factvar2 = numpy.arange(8, dtype=float) * 2

        fmean1 = xfunc_mean(factvar1)
        fmean2 = xfunc_mean(factvar2)

        groups = group_xfuncs([fmean1, fmean2])

        assert len(groups) == 2
        assert len(groups[0]) == 1
        assert len(groups[1]) == 1

    def test_group_mixed_compatible_and_independent(self):
        """Mix of compatible and independent xfuncs should create correct groups."""
        factvar1 = numpy.arange(8, dtype=float)
        factvar2 = numpy.arange(8, dtype=float) * 2

        fmean1 = xfunc_mean(factvar1)
        fstddev1 = xfunc_stddev(factvar1)
        fmean2 = xfunc_mean(factvar2)

        groups = group_xfuncs([fmean1, fstddev1, fmean2])

        assert len(groups) == 2
        # First group: mean and stddev with factvar1
        assert len(groups[0]) == 2
        # Second group: mean with factvar2
        assert len(groups[1]) == 1


class TestCompositeXfuncCalculation:
    """Test that composite xfuncs compute correct results."""

    def test_cube_calculate_mean_and_stddev(self):
        """cube.calculate with mean and stddev should produce correct results.
        
        The user interface is unchanged: just call cube.calculate with the xfuncs.
        Internally, compatible xfuncs are automatically grouped and share
        intermediate computations for better performance.
        """
        arr1 = [1, 0, 1, 0, 0, 0, 0, 1]
        factvar = numpy.arange(8, dtype=float)

        cube = xcube([arr1])

        # User just calls calculate with the xfuncs they need
        fmean = xfunc_mean(factvar)
        fstddev = xfunc_stddev(factvar)
        means, stddevs = cube.calculate([fmean, fstddev])

        # Verify results are correct
        # For arr1=[1,0,1,0,0,0,0,1]: category 0 has rows [1,3,4,5,6], category 1 has rows [0,2,7]
        # factvar=[0,1,2,3,4,5,6,7]
        # Category 0: values [1,3,4,5,6] -> mean=3.8, std=1.9235...
        # Category 1: values [0,2,7] -> mean=3.0, std=3.6055...
        expected_means = [3.8, 3.0]
        expected_stddevs = [numpy.std([1, 3, 4, 5, 6], ddof=1), numpy.std([0, 2, 7], ddof=1)]

        assert arr_eq(means, expected_means)
        assert arr_eq(stddevs, expected_stddevs)

        # checking that the value of the stddev calculated
        # alone is the same as the one with the mean
        stddevs_alone = cube.calculate([fstddev])
        assert arr_eq(stddevs_alone, expected_stddevs)

    def test_cube_calculate_mean_and_stddev_2d(self):
        """cube.calculate with mean and stddev works with 2D dimensions.
        
        The interface is the same regardless of dimension complexity.
        """
        arr1 = [1, 0, 1, 0, 0, 0, 0, 1]
        arr2 = [[1, 0], [0, 0], [1, 0], [0, 1], [0, 1], [1, 0], [0, 0], [0, 0]]
        factvar = numpy.arange(8, dtype=float)

        cube = xcube([arr1, arr2])

        # User just calls calculate
        fmean = xfunc_mean(factvar)
        fstddev = xfunc_stddev(factvar)
        means, stddevs = cube.calculate([fmean, fstddev])

        # Expected values (computed manually for this data)
        expected_means = [
            [[3.5, 5.0], [7.0, 1.0]],
            [[4.0, 3.5], [3.0, float("nan")]],
        ]
        expected_stddevs = [
            [
                [numpy.std([1, 3, 4, 6], ddof=1), float("nan")],
                [float("nan"), numpy.std([0, 2], ddof=1)],
            ],
            [
                [numpy.std([1, 5, 6], ddof=1), numpy.std([3, 4], ddof=1)],
                [numpy.std([0, 2, 7], ddof=1), float("nan")],
            ],
        ]

        assert arr_eq(means, expected_means)
        assert arr_eq(stddevs, expected_stddevs)


class TestXcubeWithCompositeXfuncs:
    """Test xcube.calculate with automatic grouping."""

    def test_calculate_auto_groups_compatible_xfuncs(self):
        """xcube.calculate should automatically group compatible xfuncs."""
        arr1 = [1, 0, 1, 0, 0, 0, 0, 1]
        arr2 = [[1, 0], [0, 0], [1, 0], [0, 1], [0, 1], [1, 0], [0, 0], [0, 0]]
        factvar = numpy.arange(8, dtype=float)

        cube = xcube([arr1, arr2])

        fmean = xfunc_mean(factvar)
        fstddev = xfunc_stddev(factvar)

        # This should internally group them and compute efficiently
        means, stddevs = cube.calculate([fmean, fstddev])

        # Expected values (from test_xcube_calculate_mean_and_stddev)
        expected_means = [
            [[3.5, 5.0], [7.0, 1.0]],
            [[4.0, 3.5], [3.0, float("nan")]],
        ]
        expected_stddevs = [
            [
                [numpy.std([1, 3, 4, 6], ddof=1), float("nan")],
                [float("nan"), numpy.std([0, 2], ddof=1)],
            ],
            [
                [numpy.std([1, 5, 6], ddof=1), numpy.std([3, 4], ddof=1)],
                [numpy.std([0, 2, 7], ddof=1), float("nan")],
            ],
        ]

        assert arr_eq(means, expected_means)
        assert arr_eq(stddevs, expected_stddevs)

    def test_calculate_mixed_groups(self):
        """xcube.calculate with mixed compatible and independent xfuncs."""
        arr1 = [1, 0, 1, 0, 0, 0, 0, 1]
        factvar1 = numpy.arange(8, dtype=float)
        factvar2 = numpy.arange(8, dtype=float) * 2

        cube = xcube([arr1])

        fmean1 = xfunc_mean(factvar1)
        fstddev1 = xfunc_stddev(factvar1)
        fmean2 = xfunc_mean(factvar2)

        # mean1 and stddev1 should be grouped, mean2 independent
        means1, stddevs1, means2 = cube.calculate([fmean1, fstddev1, fmean2])

        # Verify results match individual calculations
        means1_expected = cube.calculate([xfunc_mean(factvar1)])[0]
        stddevs1_expected = cube.calculate([xfunc_stddev(factvar1)])[0]
        means2_expected = cube.calculate([xfunc_mean(factvar2)])[0]

        assert arr_eq(means1, means1_expected)
        assert arr_eq(stddevs1, stddevs1_expected)
        assert arr_eq(means2, means2_expected)

    def test_calculate_uses_composite_for_compatible_xfuncs(self):
        """Verify that xcube.calculate actually uses CompositeXfunc for compatible xfuncs."""
        arr1 = [1, 0, 1, 0, 0, 0, 0, 1]
        factvar = numpy.arange(8, dtype=float)

        cube = xcube([arr1])

        fmean = xfunc_mean(factvar)
        fstddev = xfunc_stddev(factvar)

        # Check that grouping creates a CompositeXfunc
        grouped_funcs, original_to_grouped = cube._group_compatible_funcs([fmean, fstddev])

        assert len(grouped_funcs) == 1, "Should create one CompositeXfunc for compatible xfuncs"
        assert isinstance(grouped_funcs[0], CompositeXfunc), "Should use CompositeXfunc"

        # Verify results are still correct
        means, stddevs = cube.calculate([fmean, fstddev])
        means_expected = cube.calculate([xfunc_mean(factvar)])[0]
        stddevs_expected = cube.calculate([xfunc_stddev(factvar)])[0]

        assert arr_eq(means, means_expected)
        assert arr_eq(stddevs, stddevs_expected)

    def test_calculate_does_not_use_composite_for_incompatible_xfuncs(self):
        """Verify that xcube.calculate does NOT use CompositeXfunc for incompatible xfuncs."""
        arr1 = [1, 0, 1, 0, 0, 0, 0, 1]
        factvar1 = numpy.arange(8, dtype=float)
        factvar2 = numpy.arange(8, dtype=float) * 2  # Different values

        cube = xcube([arr1])

        fmean1 = xfunc_mean(factvar1)
        fmean2 = xfunc_mean(factvar2)

        # Check that grouping does NOT create a CompositeXfunc
        grouped_funcs, original_to_grouped = cube._group_compatible_funcs([fmean1, fmean2])

        assert len(grouped_funcs) == 2, "Should keep incompatible xfuncs separate"
        assert not isinstance(grouped_funcs[0], CompositeXfunc), "Should NOT use CompositeXfunc for first"
        assert not isinstance(grouped_funcs[1], CompositeXfunc), "Should NOT use CompositeXfunc for second"

        # Verify the original xfuncs are used directly
        assert grouped_funcs[0] is fmean1, "Should use original xfunc directly"
        assert grouped_funcs[1] is fmean2, "Should use original xfunc directly"

        # Verify results are still correct
        means1, means2 = cube.calculate([fmean1, fmean2])
        means1_expected = cube.calculate([xfunc_mean(factvar1)])[0]
        means2_expected = cube.calculate([xfunc_mean(factvar2)])[0]

        assert arr_eq(means1, means1_expected)
        assert arr_eq(means2, means2_expected)

    def test_calculate_does_not_use_composite_for_single_xfunc(self):
        """Verify that xcube.calculate does NOT use CompositeXfunc for a single xfunc."""
        arr1 = [1, 0, 1, 0, 0, 0, 0, 1]
        factvar = numpy.arange(8, dtype=float)

        cube = xcube([arr1])

        fmean = xfunc_mean(factvar)

        # Check that grouping does NOT create a CompositeXfunc for single xfunc
        grouped_funcs, original_to_grouped = cube._group_compatible_funcs([fmean])

        assert len(grouped_funcs) == 1, "Should have one func"
        assert not isinstance(grouped_funcs[0], CompositeXfunc), "Should NOT use CompositeXfunc for single xfunc"
        assert grouped_funcs[0] is fmean, "Should use original xfunc directly"

    def test_calculate_mixed_compatible_and_incompatible(self):
        """Verify correct grouping with a mix of compatible and incompatible xfuncs."""
        arr1 = [1, 0, 1, 0, 0, 0, 0, 1]
        factvar1 = numpy.arange(8, dtype=float)
        factvar2 = numpy.arange(8, dtype=float) * 2

        cube = xcube([arr1])

        # Two compatible (same factvar) and one incompatible (different factvar)
        fmean1 = xfunc_mean(factvar1)
        fstddev1 = xfunc_stddev(factvar1)
        fmean2 = xfunc_mean(factvar2)

        grouped_funcs, original_to_grouped = cube._group_compatible_funcs([fmean1, fstddev1, fmean2])

        # Should have 2 groups: one CompositeXfunc for (mean1, stddev1), one regular for mean2
        assert len(grouped_funcs) == 2, "Should have 2 groups"
        assert isinstance(grouped_funcs[0], CompositeXfunc), "First group should be CompositeXfunc"
        assert not isinstance(grouped_funcs[1], CompositeXfunc), "Second group should NOT be CompositeXfunc"
        assert grouped_funcs[1] is fmean2, "Second group should be the original xfunc"

        # Verify mapping is correct
        assert original_to_grouped[0] == (0, 0), "fmean1 should map to CompositeXfunc index 0"
        assert original_to_grouped[1] == (0, 1), "fstddev1 should map to CompositeXfunc index 1"
        assert original_to_grouped[2] == (1, None), "fmean2 should map directly (no sub-index)"
