import operator
import sys
from functools import reduce

import numpy
import pytest

from catii import xcube, xfuncs

from .. import arr_eq

MAXFLOAT = sys.float_info.max
arr1 = [1, 0, 1, 0, 0]  # iindex({(1,): [0, 2]}, 0, (5,))
arr2 = [1, 0, 0, 1, 0]  # iindex({(1,): [0, 3]}, 0, (5,))
wt = numpy.array([0.25, 0.3, 0.99, 1.0, float("nan")])

# The cube of arr1 x arr2 has rowids:
#      ___arr 2___
#      0       1
# a|0  [1, 4]  [3]
# r|
# r|
# 1|1  [2]     [0]
#
# and therefore weights:
#      ______arr 2_______
#      0           1
# a|0  [0.3, nan]  [1.0]
# r|
# r|
# 1|1  [0.99       [0.25]


class TestXfuncQuantileWorkflow:
    def _test_unweighted_workflow(self, factvar):
        # Other tests often simply test output; here we test intermediate results.
        f = xfuncs.xfunc_quantile(factvar, 0.5, ignore_missing=True)

        cube = xcube([])
        (qs,) = f.get_initial_regions(cube)
        assert arr_eq(qs, [float("nan")])
        coordinates = None
        f.fill(coordinates, (qs,))
        assert qs.tolist() == [3.0]
        assert f.reduce(cube, (qs,)) == [3.0]

        cube = xcube([arr1])
        (qs,) = f.get_initial_regions(cube)
        assert arr_eq(qs, [float("nan"), float("nan")])
        coordinates = reduce(operator.add, cube.strided_dims())
        f.fill(coordinates, (qs,))
        assert qs.tolist() == [4.0, 1.0]
        assert f.reduce(cube, (qs,)).tolist() == [4.0, 1.0]

        cube = xcube([arr1, arr2])
        (qs,) = f.get_initial_regions(cube)
        assert arr_eq(qs, [[float("nan"), float("nan")], [float("nan"), float("nan")]])
        coordinates = reduce(operator.add, cube.strided_dims())
        f.fill(coordinates, (qs,))
        assert arr_eq(qs, [[3.5, 4.0], [float("nan"), 1.0]])
        assert arr_eq(f.reduce(cube, (qs,)), [[3.5, 4.0], [float("nan"), 1.0]])

    def _test_weighted_workflow(self, factvar):
        # Other tests often simply test output; here we test intermediate results.
        f = xfuncs.xfunc_quantile(factvar, 0.5, weights=wt, ignore_missing=True)

        cube = xcube([arr1, arr2])
        (qs,) = f.get_initial_regions(cube)
        assert arr_eq(qs, [[float("nan"), float("nan")], [float("nan"), float("nan")]])
        coordinates = reduce(operator.add, cube.strided_dims())
        f.fill(coordinates, (qs,))
        assert arr_eq(qs, [[2.0, 4.0], [float("nan"), 1.0]])
        assert arr_eq(f.reduce(cube, (qs,)), [[2.0, 4.0], [float("nan"), 1.0]])

    def test_single_arr_with_nan_workflow(self):
        factvar = [1.0, 2.0, float("nan"), 4.0, 5.0]
        self._test_unweighted_workflow(factvar)
        self._test_weighted_workflow(factvar)

    def test_tuple_arr_workflow(self):
        factvar = ([1.0, 2.0, 3.0, 4.0, 5.0], [True, True, False, True, True])
        self._test_unweighted_workflow(factvar)
        self._test_weighted_workflow(factvar)

    def test_tuple_arr_with_nan_workflow(self):
        factvar = ([1.0, 2.0, float("nan"), 4.0, 5.0], [True, True, False, True, True])
        self._test_unweighted_workflow(factvar)
        self._test_weighted_workflow(factvar)

    def test_tuple_arr_integer_workflow(self):
        factvar = ([1, 2, 3, 4, 5], [True, True, False, True, True])
        self._test_unweighted_workflow(factvar)
        self._test_weighted_workflow(factvar)


class TestXfuncQuantileWeights:
    def test_weights(self):
        factvar = [1.0, 2.0, 3.0, 4.0, 5.0]

        qs = xcube([arr1]).quantile(factvar, 0.5, weights=None)
        assert arr_eq(qs, [4.0, 2.0])

        qs = xcube([arr1]).quantile(factvar, 0.5, weights=[0.25, 0.3, 0.99, 1.0, 0.5])
        assert arr_eq(qs, [3.2, 1.74747475])

        qs = xcube([arr1]).quantile(
            factvar, 0.5, weights=[0.25, 0.3, 0.99, 1.0, float("nan")]
        )
        assert arr_eq(qs, [float("nan"), 1.74747475])

        qs = xcube([arr1]).quantile(
            factvar,
            0.5,
            weights=[0.25, 0.3, 0.99, 1.0, float("nan")],
            ignore_missing=True,
        )
        assert arr_eq(qs, [2.7, 1.74747475])

        qs = xcube([arr1]).quantile(factvar, 0.5, weights=0.5)
        assert arr_eq(qs, [3.0, 1.0])

        qs = xcube([arr1]).quantile(factvar, 0.5, weights=float("nan"))
        assert arr_eq(qs, [float("nan"), float("nan")])

        qs = xcube([arr1]).quantile(
            factvar, 0.5, weights=float("nan"), ignore_missing=True
        )
        assert arr_eq(qs, [float("nan"), float("nan")])


class TestXfuncQuantileMissingness:
    # Make sure each combination of these options works properly:
    #  * ignore_missing = True/False
    #  * return_missing_as = NaN/(<sentinel value>, False)
    #  * at least one output cell which has no inputs
    #  * fact variable (with missings) as:
    #    * single array with NaN
    #    * no NaN, but separate "validity" array
    #    * includes NaN, but separate "validity" array
    #  * weights variable (with missings) as:
    #    * single array with NaN
    #    * no NaN, but separate "validity" array
    #    * includes NaN, but separate "validity" array
    arr_with_empty_cell = [0, 0, 0, 1, 0]

    dirty_fact = [1.0, 2.0, float("nan"), 4.0, 5.0]
    clean_fact = [1.0, 2.0, MAXFLOAT, 4.0, 5.0]
    fact_validity = [True, True, False, True, True]
    fact_all_valid = [True] * 5

    dirty_weights = [9.0, 9.0, float("nan"), 9.0, 9.0]
    clean_weights = [9.0, 9.0, MAXFLOAT, 9.0, 9.0]
    weights_validity = [True, True, False, True, True]

    params = [
        [dirty_fact, None],
        [(clean_fact, fact_validity), None],
        [(dirty_fact, fact_validity), None],
        [dirty_fact, dirty_weights],
        [(clean_fact, fact_validity), dirty_weights],
        [(dirty_fact, fact_validity), dirty_weights],
        [(clean_fact, fact_all_valid), dirty_weights],
        [dirty_fact, (clean_weights, weights_validity)],
        [(clean_fact, fact_validity), (clean_weights, weights_validity)],
        [(dirty_fact, fact_validity), (clean_weights, weights_validity)],
        [(clean_fact, fact_all_valid), (clean_weights, weights_validity)],
        [dirty_fact, (dirty_weights, weights_validity)],
        [(clean_fact, fact_validity), (dirty_weights, weights_validity)],
        [(dirty_fact, fact_validity), (dirty_weights, weights_validity)],
        [(clean_fact, fact_all_valid), (dirty_weights, weights_validity)],
    ]

    @pytest.mark.parametrize("factvar,weights", params)
    def test_propagate_missing_return_nan(self, factvar, weights):
        # The cube of arr1 has rowids:
        # 0           1
        # [1, 3, 4]  [0, 2]
        # and fact values:
        # 0           1
        # [2.0, 4.0, 5.0]  [1.0, nan]
        qs = xcube([arr1]).quantile(factvar, 0.5, weights)
        # Cell (1,) MUST be missing, because it had a missing input.
        assert arr_eq(qs, [4.0 if weights is None else 3.0, float("nan")])

        # The cube of arr1 x arr_with_empty_cell has fact values:
        #      _______idx w_______
        #      0           1
        # i|0  [2.0, 5.0]  [4.0]
        # d|
        # x|
        # 1|1  [1.0, nan]       []
        qs = xcube([arr1, self.arr_with_empty_cell]).quantile(factvar, 0.5, weights)
        # Cell (1, 0) MUST be missing, because it had a missing input.
        # Cell (1, 1) MUST be missing, because it had no inputs.
        assert arr_eq(
            qs, [[3.5 if weights is None else 2.0, 4.0], [float("nan"), float("nan")]]
        )

    @pytest.mark.parametrize("factvar,weights", params)
    def test_ignore_missing_return_nan(self, factvar, weights):
        # The cube of arr1 has rowids:
        # 0           1
        # [1, 3, 4]  [0, 2]
        # and fact values:
        # 0           1
        # [2.0, 4.0, 5.0]  [1.0, nan]
        qs = xcube([arr1]).quantile(factvar, 0.5, weights, ignore_missing=True)
        # Cell (1,) MUST NOT be missing, because it had a valid input.
        assert arr_eq(qs, [4.0 if weights is None else 3.0, 1.0])

        # The cube of arr1 x arr_with_empty_cell has fact values:
        #      _______idx w_______
        #      0           1
        # i|0  [2.0, 5.0]  [4.0]
        # d|
        # x|
        # 1|1  [1.0, nan]       []
        qs = xcube([arr1, self.arr_with_empty_cell]).quantile(
            factvar, 0.5, weights, ignore_missing=True
        )
        # Cell (1, 0) MUST NOT be missing, because it had a valid input.
        # Cell (1, 1) MUST be missing, because it had no inputs.
        assert arr_eq(qs, [[3.5 if weights is None else 2.0, 4.0], [1.0, float("nan")]])

    @pytest.mark.parametrize("factvar,weights", params)
    def test_propagate_missing_return_validity(self, factvar, weights):
        # The cube of arr1 has rowids:
        # 0           1
        # [1, 3, 4]  [0, 2]
        # and fact values:
        # 0           1
        # [2.0, 4.0, 5.0]  [1.0, nan]
        qs, validity = xcube([arr1]).quantile(
            factvar, 0.5, weights, return_missing_as=(-1, False)
        )
        # Cell (1,) MUST be missing, because it had a missing input.
        assert arr_eq(qs, [4.0 if weights is None else 3.0, -1])
        assert arr_eq(validity, [True, False])

        # The cube of arr1 x arr_with_empty_cell has fact values:
        #      _______idx w_______
        #      0           1
        # i|0  [2.0, 5.0]  [4.0]
        # d|
        # x|
        # 1|1  [1.0, nan]       []
        qs, validity = xcube([arr1, self.arr_with_empty_cell]).quantile(
            factvar, 0.5, weights, return_missing_as=(-1, False)
        )
        # Cell (1, 0) MUST be missing, because it had a missing input.
        # Cell (1, 1) MUST be missing, because it had no inputs.
        assert arr_eq(qs, [[3.5 if weights is None else 2.0, 4.0], [-1, -1]])
        assert arr_eq(validity, [[True, True], [False, False]])

    @pytest.mark.parametrize("factvar,weights", params)
    def test_ignore_missing_return_validity(self, factvar, weights):
        # The cube of arr1 has rowids:
        # 0           1
        # [1, 3, 4]  [0, 2]
        # and fact values:
        # 0           1
        # [2.0, 4.0, 5.0]  [1.0, nan]
        qs, validity = xcube([arr1]).quantile(
            factvar, 0.5, weights, ignore_missing=True, return_missing_as=(-1, False)
        )
        # Cell (1,) MUST NOT be missing, because it had a valid input.
        assert arr_eq(qs, [4.0 if weights is None else 3.0, 1.0])
        assert arr_eq(validity, [True, True])

        # The cube of arr1 x arr_with_empty_cell has fact values:
        #      _______idx w_______
        #      0           1
        # i|0  [2.0, 5.0]  [4.0]
        # d|
        # x|
        # 1|1  [1.0, nan]       []
        qs, validity = xcube([arr1, self.arr_with_empty_cell]).quantile(
            factvar, 0.5, weights, ignore_missing=True, return_missing_as=(-1, False)
        )
        # Cell (1, 0) MUST NOT be missing, because it had a valid input.
        # Cell (1, 1) MUST be missing, because it had no inputs.
        assert arr_eq(qs, [[3.5 if weights is None else 2.0, 4.0], [1.0, -1]])
        assert arr_eq(validity, [[True, True], [True, False]])


class TestXfuncQuantileArgsortTieOrder:
    """Reproduces the weighted-quantile non-determinism across numpy versions.

    weighted_quantile_1d sorts fact values with a bare ``a.argsort()``, whose
    order for TIED values is unspecified. The parallel weights are reordered to
    match, so when equal values carry different weights the tie order changes
    which weight lands at the quantile boundary -> a different (but still valid)
    interpolated result.

    numpy 2.4.0 retired the separate heapsort implementation and mapped the
    'heapsort' kind onto quicksort, which changed the tie order vs numpy 1.x.
    On the data below (many tied 60.0 / 70.0 values with differing weights) the
    0.25 quantile therefore differs:

        numpy 1.x : 58.56617647058823   (the historical / expected value)
        numpy 2.x : 55.88383838383838

    This test asserts the numpy-1 value, so it PASSES under numpy 1 and FAILS
    under numpy 2 - making the regression visible in the CI matrix. The fix is
    a deterministic tie-break in weighted_quantile_1d:

        ind = numpy.lexsort((w, a))   # sort by value, break ties by weight

    which yields 58.566... on both numpy versions (verified). Once that lands,
    this test passes everywhere and the note can be removed.
    """

    # Single-cell cube (all rows in coordinate 0).
    idx = [0] * 20
    fact = [
        90.0,
        float("nan"),
        float("nan"),
        float("nan"),
        60.0,
        40.0,
        50.0,
        60.0,
        60.0,
        70.0,
        70.0,
        70.0,
        70.0,
        50.0,
        70.0,
        30.0,
        60.0,
        40.0,
        70.0,
        10.0,
    ]
    weights = [
        0.0,
        0.7366666666666667,
        0.0,
        1.0725,
        1.0725,
        0.0,
        0.0,
        1.43,
        0.7366666666666667,
        1.0725,
        0.0,
        0.7366666666666667,
        0.7366666666666667,
        0.7366666666666667,
        1.0725,
        0.0,
        0.7366666666666667,
        0.0,
        1.43,
        1.43,
    ]

    def test_weighted_quantile_tie_order_is_deterministic(self):
        qs = xcube([self.idx]).quantile(
            self.fact, 0.25, self.weights, ignore_missing=True
        )
        # The historically-correct (numpy 1.x) value. Under numpy 2.x the
        # unstable argsort tie-order currently yields 55.88383838383838 instead.
        assert arr_eq(qs, [58.56617647058823])
