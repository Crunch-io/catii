import re

import numpy
import pytest

from catii import iindex
from catii.iindexes import column_stack, fit_dtype


def pytest_raises(exctype, match=None, msg=None):
    if match is None and msg is not None:
        match = re.escape(msg)
    return pytest.raises(exctype, match=match)


class TestIndexCreation:
    def test_direct_construction(self):
        idx = iindex({(1,): [0, 2, 7]}, 0, (8,))
        assert idx.to_dict() == {(1,): [0, 2, 7]}
        assert idx.common == 0
        assert idx.shape == (8,)

    def test_type_errors(self):
        with pytest_raises(
            TypeError, msg="iindex.shape MUST be of type 'tuple', not 8."
        ):
            iindex({}, 0, 8)

        with pytest_raises(TypeError, msg="Index coordinates MUST all be tuples."):
            iindex({1: [0, 2, 7]}, 0, (8,))

        with pytest_raises(
            TypeError,
            msg="Index array dtype is dtype('int64'), but expected dtype('uint32').",
        ):
            iindex({(1,): numpy.array([0, 2, 7], dtype=int)}, 0, (8,))

        with pytest_raises(
            TypeError, msg="Index[(1,)] could not be converted to an array.",
        ):
            iindex({(1,): object()}, 0, (8,))


class TestValidate:
    def test_type_errors(self):
        with pytest_raises(
            ValueError,
            msg="Found index shape with wrong types: {<class 'int'>, <class 'str'>}.",
        ):
            iindex({}, 0, ("a", 8)).validate()

        with pytest_raises(ValueError, msg="Index[(1,)] contains NumPy coordinate 1."):
            iindex({(numpy.array([1])[0],): [2]}, 0, (5,)).validate()

        with pytest_raises(ValueError, msg="Index[(0,)] contains common value 0."):
            iindex({(0,): [2]}, 0, (5,)).validate()

        with pytest_raises(
            ValueError, msg="Index[(1,)] is of dtype: float64 but should be uint32."
        ):
            ii = iindex({}, 0, (5,))
            ii[(1,)] = numpy.array([2], dtype=float)
            ii.validate()

        with pytest_raises(ValueError, msg="Index[(1,)] is not a numpy array."):
            ii = iindex({}, 0, (5,))
            ii[(1,)] = "abc"
            ii.validate()

        with pytest_raises(ValueError, msg="Index[(1,)] is not unique."):
            iindex({(1,): [2, 3, 4, 3]}, 0, (5,)).validate()

        with pytest_raises(ValueError, msg="Index[(1,)] is not sorted."):
            iindex({(1,): [4, 3, 2]}, 0, (5,)).validate()

        with pytest_raises(
            ValueError, msg="Index[(1,)] and [(2,)] contain the same rowids [2, 4]."
        ):
            iindex({(1,): [2, 3, 4], (2,): [1, 2, 4]}, 0, (5,)).validate(
                check_comprehensive_unique=True
            )


class TestEquality:
    def test_unequal_indexes(self):
        idx = iindex({(99,): [1, 3, 5], (88,): [2, 4, 6]}, common=0, shape=(10,))

        # Two indexes that contain the same information MUST NOT
        # compare equal if their common value differs.
        assert idx != iindex(
            {(0,): [0, 7, 8, 9], (99,): [1, 3, 5], (88,): [2, 4, 6]},
            common=11,
            shape=(10,),
        )

    def test_equal_indexes(self):
        idx = iindex({(99,): [1, 3, 5], (88,): [2, 4, 6]}, common=0, shape=(10,))

        assert idx == iindex(
            {(99,): [1, 3, 5], (88,): [2, 4, 6]}, common=0, shape=(10,),
        )

        assert (idx == {3: "hi"}) is False

        assert idx.nbytes > 0


class TestStr:
    def test_str(self):
        # An index with few entries should print in full.
        idx = iindex({(99,): [1, 3, 5], (88,): [2, 4, 6]}, common=0, shape=(10,))
        assert str(idx) == (
            "iindex(shape=(10,), common=0, entries={"
            "(88,): array([2, 4, 6], dtype=uint32), "
            "(99,): array([1, 3, 5], dtype=uint32)})"
        )

        # An index with many entries (>10) should be summarized.
        idx = iindex(
            {(x,): [x, x + 1] for x in range(1, 23, 2)}, common=0, shape=(30,),
        )
        assert "..." in str(idx)
        assert str(idx).startswith("iindex(shape=(30,), common=0, entries={")
        # There MUST be 3 entries on either side of the "..."
        assert str(idx).count("array") == 3 * 2


class TestProperties:
    def test_props(self):
        idx = iindex.from_array([1, 2, 1, 2])
        assert idx.abscissae == {1, 2}
        assert idx.sparsity == 50  # 50% of the values have been omitted
        assert idx.size == 4  # there are 4 entries total.
        assert idx.shape == (4,)
        assert idx.ndim == 1

        idx = iindex({}, common=0, shape=(0,))
        assert idx.sparsity == 0
        assert idx.abscissae == set()


class TestShiftCommon:
    def test_shift_common(self):
        idx = iindex({(99,): [1, 3, 5], (88,): [2, 4, 6]}, common=0, shape=(10,))
        idx.shift_common()

        assert idx.common == 0

        idx[(99,)] = numpy.concatenate((idx[(99,)], [7, 8, 9]))

        idx.shift_common()
        assert idx.common == 99

        assert idx.common_rowids(0).tolist() == [1, 3, 5, 7, 8, 9]

    def test_shift_common2d(self):
        idx = iindex({(99, 0): [1, 3, 4], (88, 1): [2, 3, 4]}, common=0, shape=(5, 2),)

        assert idx.common == 0
        assert numpy.array_equal(
            idx.to_array(), [[0, 0], [99, 0], [0, 88], [99, 88], [99, 88]]
        )

        idx[(99, 1)] = numpy.array([1])
        idx.shift_common()

        assert idx.common == 99
        assert numpy.array_equal(
            idx.to_array(), [[0, 0], [99, 99], [0, 88], [99, 88], [99, 88]]
        )

        assert idx.common_rowids().tolist() == [0, 1, 2, 3, 4]
        assert idx.common_rowids(0).tolist() == [1, 3, 4]
        assert idx.common_rowids(1).tolist() == [1]

    def test_shift_common_explicit(self):
        idx = iindex({(99,): [1, 3, 5], (88,): [2, 4, 6]}, common=0, shape=(10,))
        idx.shift_common(99)

        assert idx.common == 99
        assert idx.to_dict() == {
            (88,): [2, 4, 6],
            (0,): [0, 7, 8, 9],
        }


class TestForce:
    def test_items_force(self):
        idx = iindex({(99,): [1, 3, 5], (88,): [2, 4, 6]}, common=0, shape=(10,))

        assert {idx: rows.tolist() for idx, rows in idx.items(force=True)} == {
            (0,): [0, 7, 8, 9],
            (88,): [2, 4, 6],
            (99,): [1, 3, 5],
        }

    def test_items_force_2d(self):
        idx = iindex({(99, 0): [1, 3, 4], (88, 1): [2, 3, 4]}, common=0, shape=(5, 2),)

        assert {idx: rows.tolist() for idx, rows in idx.items(force=True)} == {
            (0, 0): [0, 2],
            (0, 1): [0, 1],
            (88, 1): [2, 3, 4],
            (99, 0): [1, 3, 4],
        }

    def test_get_force(self):
        idx = iindex({(99,): [1, 3, 5], (88,): [2, 4, 6]}, common=0, shape=(10,))
        assert idx.get((0,)) is None
        assert idx.get((0,), force=True).tolist() == [0, 7, 8, 9]


class TestSetIf:
    def test_set_if(self):
        idx = iindex({}, 0, (10,))
        idx.set_if((1,), [1, 2])
        assert idx.to_dict() == {(1,): [1, 2]}
        idx.set_if((2,), [])
        assert idx.to_dict() == {(1,): [1, 2]}
        idx.set_if((2,), None)
        assert idx.to_dict() == {(1,): [1, 2]}

    def test_set_if_copy(self):
        idx = iindex({}, 0, (10,))
        arr = numpy.array([1, 2], dtype=numpy.uint32)
        idx.set_if((1,), arr)
        assert idx[(1,)] is not arr
        idx.set_if((1,), arr, copy=False)
        assert idx[(1,)] is arr


class TestSetOpMethods:
    def test_union_update(self):
        idx = iindex({(1,): [1, 3]}, 0, (10,))
        idx.union_update({(1,): [2, 4], (2,): [5]})
        assert idx.to_dict() == {(1,): [1, 2, 3, 4], (2,): [5]}

    def test_intersection_update(self):
        idx = iindex({(1,): [1, 3], (3,): [9]}, 0, (10,))
        idx.intersection_update({(1,): [2, 3], (2,): [5]})
        assert idx.to_dict() == {(1,): [3]}

    def test_difference_update(self):
        idx = iindex({(1,): [1, 3, 5, 7], (3,): [9]}, 0, (10,))
        idx.difference_update({(1,): [3, 7], (2,): [5]})
        assert idx.to_dict() == {(1,): [1, 5], (3,): [9]}


class TestCopy:
    def test_copy(self):
        idx = iindex({(99,): [1, 3, 5], (88,): [2, 4, 6]}, common=0, shape=(10,))
        assert idx.copy() == idx


class TestReindexed:
    def test_reindexed(self):
        idx = iindex({(99,): [1, 3, 5], (88,): [2, 4, 6]}, common=0, shape=(10,))
        assert idx.reindexed({99: 1, 88: 2}).to_dict() == {
            (1,): [1, 3, 5],
            (2,): [2, 4, 6],
        }

    def test_reindex_direct_mapping(self):
        idx = iindex({(99,): [1, 3, 5], (88,): [2, 4, 6]}, common=0, shape=(10,))
        assert idx.reindexed({99: 1, 88: 2}).to_dict() == {
            (1,): [1, 3, 5],
            (2,): [2, 4, 6],
        }

    def test_reindex_dupe_common(self):
        idx = iindex({(99,): [1, 3, 5], (88,): [2, 4, 6]}, common=0, shape=(10,))

        # Note there are two mappings that point to the same final value;
        # they both become common because they are now the most numerous.
        reindexed = idx.reindexed({99: 1, 88: 2, 0: 2})
        assert reindexed.common == 2
        assert reindexed.to_dict() == {(1,): [1, 3, 5]}

    def test_reindex_dupes(self):
        idx = iindex({(99,): [1, 3, 5], (88,): [2, 4, 6]}, common=0, shape=(10,))

        # Note there are two mappings that point to the same final value;
        # they both become common because they are now the most numerous.
        reindexed = idx.reindexed({99: 2, 88: 2})
        assert reindexed.common == 2
        # ...and what had been the most common value is now explicit.
        assert reindexed.to_dict() == {(0,): [0, 7, 8, 9]}


class TestTransformMethods:
    def test_filtered(self):
        idx = iindex({(99,): [1, 3, 5], (88,): [2, 4, 6]}, common=0, shape=(10,))

        # We filtered out the odd rows, selecting 88 from rows 2, 4, 6
        # and 0 from rows 0 and 8.
        # The rectangular result would be [0, 88, 88, 88, 0].
        assert idx.filtered(
            numpy.array([True if v % 2 == 0 else False for v in range(10)]), 5
        ) == iindex(
            # The indexed result SHOULD NOT include 99 because it has
            # no more rowids, but it SHOULD shift the common value to 88
            # because it is now the most numerous.
            {(0,): [0, 4]},
            common=88,
            # ...and the final shape MUST have the number of matched rows,
            # not the unfiltered number of rows.
            shape=(5,),
        )

    def test_sliced(self):
        idx = iindex(
            {(1, 1): [2], (2, 0): [1, 2], (3, 1): [0], (4, 2): [0], (5, 3): [1]},
            common=0,
            shape=(10, 4),
        )

        assert idx.sliced(1).to_dict() == {
            (1,): [2],
            (3,): [0],
        }
        assert idx.sliced([2, 0, 1]).to_dict() == {
            (1, 2): [2],
            (2, 1): [1, 2],
            (3, 2): [0],
            (4, 0): [0],
            # dropped because (,3) is not included in the order.
            # (5, 3): [1],
        }

        idx2 = iindex({(99, 0): [1, 3, 4], (88, 1): [1, 2, 4]}, common=0, shape=(5, 2),)
        # col 3 doesn't exist, so will be correctly ignored.
        assert idx2.sliced([0, 3]).to_dict() == {(99, 0): [1, 3, 4]}

        idx3 = iindex(
            {(1, 1, 1): [2], (2, 0, 1): [1, 2], (3, 1, 0): [0]},
            common=0,
            shape=(10, 4, 2),
        )
        assert idx3.sliced(None, 1).to_dict() == {
            (1, 1): [2],
            (2, 0): [1, 2],
        }

    def test_sliced_multiaxis(self):
        idx = iindex(
            {
                (1, 1, 0): [2],
                (2, 0, 0): [1, 2],
                (3, 1, 1): [0],
                (4, 2, 1): [0],
                (5, 3, 1): [1],
            },
            common=0,
            shape=(10, 4, 2),
        )

        assert idx.sliced([2, 0, 1], [1, 0]).to_dict() == {
            (1, 2, 1): [2],
            (2, 1, 1): [1, 2],
            (3, 2, 0): [0],
            (4, 0, 0): [0],
            # dropped because (,3) is not included in the order.
            # (5, 3): [1],
        }

    def test_sliced_multiaxis_None(self):
        idx = iindex(
            {
                (1, 1, 0): [2],
                (2, 0, 0): [1, 2],
                (3, 1, 1): [0],
                (4, 2, 1): [0],
                (5, 3, 1): [1],
            },
            common=0,
            shape=(10, 4, 2),
        )

        assert idx.sliced(None, [1, 0]).to_dict() == {
            (1, 1, 1): [2],
            (2, 0, 1): [1, 2],
            (3, 1, 0): [0],
            (4, 2, 0): [0],
            (5, 3, 0): [1],
        }

    def test_sliced_drop_axis(self):
        idx = iindex({(99, 0): [1, 3, 4], (88, 1): [1, 2, 4]}, common=0, shape=(5, 2),)

        # only retain column 1. By passing 1 instead of [1], we are instructing
        # sliced to drop the 2nd axis altogether.
        assert idx.sliced(1).to_dict() == {(88,): [1, 2, 4]}

    def test_sliced_axes_mismatch(self):
        idx = iindex({(99,): [1, 3, 5], (88,): [2, 4, 6]}, common=0, shape=(10,))

        with pytest_raises(TypeError):
            idx.sliced([0, 3])

    def test_slices1d(self):
        # 1-D
        idx = iindex({(1,): [1, 3, 5], (2,): [2, 4, 6]}, common=0, shape=(10,))
        assert [(coords, ii.to_dict()) for coords, ii in idx.slices1d()] == [
            ((), {(1,): [1, 3, 5], (2,): [2, 4, 6]})
        ]

        # 2-D
        idx = iindex(
            {(1, 0): [1, 3], (2, 0): [4, 6], (1, 1): [5]}, common=0, shape=(10, 2)
        )
        assert [(coords, ii.to_dict()) for coords, ii in idx.slices1d()] == [
            ((0,), {(1,): [1, 3], (2,): [4, 6]}),
            ((1,), {(1,): [5]}),
        ]

        # 3-D
        idx = iindex(
            {(1, 0, 0): [1, 3], (2, 0, 0): [4, 6], (1, 0, 1): [5]},
            common=0,
            shape=(10, 2, 2),
        )
        assert [(coords, ii.to_dict()) for coords, ii in idx.slices1d()] == [
            ((0, 0), {(1,): [1, 3], (2,): [4, 6]}),
            # Important empty slice to allow aggregations to count 0.
            ((1, 0), {}),
            ((0, 1), {(1,): [5]}),
            ((1, 1), {}),
        ]

        # 0-D
        idx = iindex({}, common=0, shape=())
        assert [(coords, ii.to_dict()) for coords, ii in idx.slices1d()] == [
            # Important empty slice to allow aggregations to count 0.
            ((), {})
        ]


class TestToFromArray:
    def test_array_conversion(self):
        idx = iindex({(99,): [1, 3, 5], (88,): [2, 4, 6]}, common=0, shape=(10,))

        arrayver = idx.to_array(dtype=numpy.uint32)
        assert numpy.array_equal(arrayver, [0, 99, 88, 99, 88, 99, 88, 0, 0, 0])

        assert iindex.from_array(arrayver) == idx

        mapped_arrayver = idx.to_array(mapping={0: 255, 99: 1, 88: 2})
        assert numpy.array_equal(
            mapped_arrayver, [255, 1, 2, 1, 2, 1, 2, 255, 255, 255]
        )

    def test_array2d_conversion(self):
        idx = iindex({(99, 0): [1, 3, 5], (88, 1): [2, 4, 6]}, common=0, shape=(10, 2),)

        arrayver = idx.to_array()

        assert numpy.array_equal(
            arrayver,
            [
                [0, 0],
                [99, 0],
                [0, 88],
                [99, 0],
                [0, 88],
                [99, 0],
                [0, 88],
                [0, 0],
                [0, 0],
                [0, 0],
            ],
        )

        assert iindex.from_array(arrayver) == idx

        mapped_arrayver = idx.to_array(mapping={0: 255, 99: 1, 88: 2})

        assert numpy.array_equal(
            mapped_arrayver,
            [
                [255, 255],
                [1, 255],
                [255, 2],
                [1, 255],
                [255, 2],
                [1, 255],
                [255, 2],
                [255, 255],
                [255, 255],
                [255, 255],
            ],
        )

    def test_from_array_with_negatives(self):
        idx = iindex.from_array([[1, 2], [1, -1]])
        assert idx == iindex(
            shape=(2, 2), common=1, entries={(2, 1): [0], (-1, 1): [1]},
        )

    def test_from_array_empty(self):
        # If no information is provided, we have to error.
        with pytest_raises(ValueError):
            idx = iindex.from_array([])

        # If a mapping is provided, however, we can make an initial guess
        # that the common value will be one of the values in that mapping.
        # Pick the lowest one for some stability.
        idx = iindex.from_array([], mapping={-1: 2, 3: 3, 99: 99})
        assert idx == iindex(shape=(0,), common=2, entries={})

    def test_from_array_all_common(self):
        # This threw RuntimeWarning: divide by zero encountered in double_scalars
        # for a while, because the "uncommon" count is zero. Fixed by correcting
        # distinct_values to only include values actually present in the input,
        # which not only avoids the divide error but should be faster, as well.
        idx = iindex.from_array([0, 0])
        assert idx == iindex(shape=(2,), common=0, entries={})

    def test_from_array_with_more_than_five_distinct_values(self):
        # This executes a code path that was previously untested from this suite
        idx = iindex.from_array([0] * 10 + [1, 2, 3, 4, 5])
        assert idx == iindex(
            shape=(15,),
            common=0,
            entries={(1,): [10], (2,): [11], (3,): [12], (4,): [13], (5,): [14]},
        )

    def test_from_array_all_common_with_explicit_counts(self):
        idx = iindex.from_array([0, 0, 0, 0, 0, 1], counts={0: 5, 1: 1})
        assert idx == iindex(shape=(6,), common=0, entries={(1,): [5]})

    def test_common_not_mapped(self):
        idx = iindex({(0,): [1, 3, 5], (1,): [0, 2, 4]}, common=2, shape=(6,))

        arrayver = idx.to_array(dtype=numpy.uint32, mapping={0: 10, 1: 11})
        assert numpy.array_equal(arrayver, [11, 10, 11, 10, 11, 10])


class TestAppend:
    def test_append(self):
        idx = iindex({}, common=0, shape=(0,))

        with pytest_raises(TypeError):
            idx.append([])

        idx.append(iindex({(99,): [1, 3], (88,): [2, 4]}, common=0, shape=(6,),))

        assert idx.to_dict(force=True) == {
            (0,): [0, 5],
            (88,): [2, 4],
            (99,): [1, 3],
        }

        idx.append(iindex({(99,): [1, 3], (88,): [2, 4]}, common=0, shape=(6,),))

        assert idx.to_dict(force=True) == {
            (0,): [0, 5, 6, 11],
            (88,): [2, 4, 8, 10],
            (99,): [1, 3, 7, 9],
        }

    def test_append_different_common(self):
        idx = iindex({}, common=1, shape=(0,))
        idx.append(iindex({(99,): [1, 3], (88,): [2, 4]}, common=0, shape=(6,),))

        assert idx.to_dict(force=True) == {
            (0,): [0, 5],
            (88,): [2, 4],
            (99,): [1, 3],
        }

        idx.append(iindex({(88,): [1, 3]}, common=99, shape=(5,),))

        assert idx.to_dict(force=True) == {
            (0,): [0, 5],
            (88,): [2, 4, 7, 9],
            (99,): [1, 3, 6, 8, 10],
        }

    def test_append2d(self):
        idx = iindex({}, common=0, shape=(0, 2))
        idx.append(
            iindex({(99, 0): [1, 3, 4], (88, 1): [1, 2, 4]}, common=0, shape=(5, 2))
        )

        assert idx.to_dict(force=True) == {
            (0, 0): [0, 2],
            (0, 1): [0, 3],
            (88, 1): [1, 2, 4],
            (99, 0): [1, 3, 4],
        }

        idx.append(
            iindex({(99, 0): [1, 3, 4], (88, 1): [1, 2, 4]}, common=0, shape=(5, 2))
        )

        assert idx.to_dict(force=True) == {
            (0, 0): [0, 2, 5, 7],
            (0, 1): [0, 3, 5, 8],
            (88, 1): [1, 2, 4, 6, 7, 9],
            (99, 0): [1, 3, 4, 6, 8, 9],
        }

    def test_append2d_different_common(self):
        idx = iindex({}, common=1, shape=(0, 2))
        idx.append(
            iindex({(99, 0): [1, 3, 4], (88, 1): [1, 2, 4]}, common=0, shape=(5, 2))
        )

        assert idx.to_dict(force=True) == {
            (0, 0): [0, 2],
            (0, 1): [0, 3],
            (88, 1): [1, 2, 4],
            (99, 0): [1, 3, 4],
        }

        idx.append(iindex({(88, 1): [1, 2, 4]}, common=99, shape=(5, 2),))

        assert idx.to_dict(force=True) == {
            (0, 0): [0, 2],
            (0, 1): [0, 3],
            (88, 1): [1, 2, 4, 6, 7, 9],
            (99, 0): [1, 3, 4, 5, 6, 7, 8, 9],
            (99, 1): [5, 8],
        }


class TestUpdate:
    def test_update(self):
        idx = iindex({}, common=0, shape=(6,))
        idx.update({(99,): [1, 3], (88,): [2, 4]})

        assert idx.to_dict(force=True) == {
            (0,): [0, 5],
            (88,): [2, 4],
            (99,): [1, 3],
        }

        idx.update({(99,): [1, 3], (88,): [2, 4]})

        assert idx.to_dict(force=True) == {
            (0,): [0, 5],
            (88,): [2, 4],
            (99,): [1, 3],
        }

    def test_update_2d(self):
        idx = iindex({}, common=0, shape=(5, 2))
        idx.update({(99, 0): [1, 3, 4], (88, 1): [1, 2, 4]})

        assert idx.to_dict(force=True) == {
            (0, 0): [0, 2],
            (0, 1): [0, 3],
            (88, 1): [1, 2, 4],
            (99, 0): [1, 3, 4],
        }

        idx.update({(99, 0): [1, 3, 4], (88, 1): [1, 2, 4]})
        assert idx.to_dict(force=True) == {
            (0, 0): [0, 2],
            (0, 1): [0, 3],
            (88, 1): [1, 2, 4],
            (99, 0): [1, 3, 4],
        }

        idx.update({(99, 0): [1, 3, 4], (88, 1): [0, 1, 2]})
        assert idx.to_dict(force=True) == {
            (0, 0): [0, 2],
            (0, 1): [3],
            (88, 1): [0, 1, 2, 4],
            (99, 0): [1, 3, 4],
        }

    def test_update_index_with_common_value_in_entries(self):
        idx = iindex({(3,): [2, 5]}, common=0, shape=(6,))
        idx.update(
            {
                (99,): [1, 3],
                # change row 2 from value 3 to value 0. row 4 is already value 0.
                (0,): [2, 4],
            }
        )
        assert idx == iindex({(3,): [5], (99,): [1, 3]}, common=0, shape=(6,))


class TestColumnStack:
    def test_column_stack_1d(self):
        idx0 = iindex({(99,): [1, 3], (88,): [2, 4]}, 0, (5,))
        idx1 = iindex({(99,): [2], (88,): [0, 4]}, 0, (5,))
        assert column_stack((idx0, idx1)) == iindex(
            {(88, 0): [2, 4], (99, 0): [1, 3], (88, 1): [0, 4], (99, 1): [2],},
            0,
            (5, 2),
        )

        assert numpy.array_equal(
            column_stack((idx0, idx1)).to_array(),
            numpy.column_stack((idx0.to_array(), idx1.to_array())),
        )

    def test_column_stack_2d(self):
        idx0 = iindex({(99, 0): [1, 3, 4], (88, 1): [1, 2, 4]}, 0, (5, 2))
        idx1 = iindex({(99, 1): [2], (88, 0): [0, 4]}, 0, (5, 2))
        assert column_stack((idx0, idx1)) == iindex(
            {
                # idx0 -> columns 0 and 1
                (99, 0): [1, 3, 4],
                (88, 1): [1, 2, 4],
                # idx1 -> columns 2 and 3
                (88, 2): [0, 4],
                (99, 3): [2],
            },
            0,
            (5, 4),
        )

        assert numpy.array_equal(
            column_stack((idx0, idx1)).to_array(),
            numpy.column_stack((idx0.to_array(), idx1.to_array())),
        )

    def test_column_stack_1d_different_commons(self):
        idx0 = iindex({(99,): [1, 3], (88,): [2, 4]}, 0, (5,))
        idx1 = iindex({(99,): [0], (88,): [0, 4]}, 2, (5,))
        # Value `2` is chosen as the new common value because it is more common.
        assert column_stack((idx0, idx1)) == iindex(
            {
                (0, 0): [0],
                (88, 0): [2, 4],
                (99, 0): [1, 3],
                (88, 1): [0, 4],
                (99, 1): [0],
            },
            2,
            (5, 2),
        )

        assert numpy.array_equal(
            column_stack((idx0, idx1)).to_array(),
            numpy.column_stack((idx0.to_array(), idx1.to_array())),
        )

    def test_column_stack_2d_different_commons(self):
        idx0 = iindex({(99, 0): [1, 3, 4], (88, 1): [1, 2, 4]}, 0, (5, 2))
        idx1 = iindex({(99, 1): [2], (88, 0): [0, 4]}, 2, (5, 2))
        assert column_stack((idx0, idx1)) == iindex(
            {
                # idx0 -> columns 0 and 1
                (0, 0): [0, 2],
                (99, 0): [1, 3, 4],
                (0, 1): [0, 3],
                (88, 1): [1, 2, 4],
                # idx1 -> columns 2 and 3
                (88, 2): [0, 4],
                (99, 3): [2],
            },
            2,
            (5, 4),
        )

        assert numpy.array_equal(
            column_stack((idx0, idx1)).to_array(),
            numpy.column_stack((idx0.to_array(), idx1.to_array())),
        )


class TestFitDtype:
    def test_fit_dtype_uint(self):
        for pwr in (8, 16, 32):
            boundary = 2 ** pwr
            for maxval in range(boundary - 1, boundary + 2):
                assert numpy.array([maxval], dtype=fit_dtype(maxval))[0] == maxval

    def test_fit_dtype_int_negatives(self):
        for pwr in (8, 16, 32):
            boundary = -(2 ** pwr)
            for minval in range(boundary - 1, boundary + 2):
                assert numpy.array([minval], dtype=fit_dtype(0, minval))[0] == minval

    def test_fit_dtype_int_positives(self):
        for pwr in (8, 16, 32):
            boundary = 2 ** pwr
            for maxval in range(boundary - 1, boundary + 2):
                assert numpy.array([maxval], dtype=fit_dtype(maxval, -1))[0] == maxval

    def test_fit_dtype(self):
        slots = {
            -(2 ** 31): numpy.int32,
            -(2 ** 31) - 1: numpy.int64,
            -(2 ** 32): numpy.int64,
            -(2 ** 15): numpy.int16,
            -(2 ** 15) - 1: numpy.int32,
            -(2 ** 16): numpy.int32,
            -(2 ** 7): numpy.int8,
            -(2 ** 7) - 1: numpy.int16,
            -(2 ** 8): numpy.int16,
            10: numpy.uint8,
            32000: numpy.uint16,
            70000: numpy.uint32,
            2 ** 32 + 10: numpy.uint64,
        }
        for v, expected in slots.items():
            actual = fit_dtype(v)
            assert actual == expected, (v, actual, expected)
