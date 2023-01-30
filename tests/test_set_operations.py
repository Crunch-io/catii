import numpy
import pytest

from catii.set_operations import difference, intersection, union


class TestIndexIntersection(object):
    def intersect(self, a, b):
        a = None if a is None else numpy.array(a, dtype=numpy.uint32)
        b = None if b is None else numpy.array(b, dtype=numpy.uint32)
        result = intersection(a, b)
        return None if result is None else result.tolist()

    def test_intersect_happy(self):
        assert self.intersect([1, 5, 7, 9, 11], [2, 5, 9, 11, 13]) == [5, 9, 11]

    def test_intersect_empty(self):
        assert self.intersect([], []) is None
        assert self.intersect([1, 2, 3], []) is None
        assert self.intersect([], [4, 5, 6]) is None
        assert self.intersect(None, None) is None
        assert self.intersect([1, 2, 3], None) is None
        assert self.intersect(None, [4, 5, 6]) is None

    def test_intersect_overlap(self):
        # No overlap
        assert self.intersect([1, 2, 3], [4, 5, 6]) is None
        assert self.intersect([4, 5, 6], [1, 2, 3]) is None

        # Overlap by 1
        assert self.intersect([1, 2, 3], [3, 4, 5]) == [3]
        assert self.intersect([3, 4, 5], [1, 2, 3]) == [3]

        # Complete overlap
        assert self.intersect([3, 4, 5], [3, 4, 5]) == [3, 4, 5]

    def test_intersect_minmax(self):
        M = (2 ** 32) - 1
        assert self.intersect([0, 100, M], [0, 102, M]) == [0, M]

    @pytest.mark.skip("Takes too long to run every time")
    def test_intersect_huge(self):
        a = numpy.arange(2 ** 30, dtype=numpy.uint32)
        b = numpy.array([1, 100], dtype=numpy.uint32)
        assert intersection(a, b).tolist() == [1, 100]

    def test_multidim_errors(self):
        with pytest.raises(ValueError):
            self.intersect([[1, 2, 3], [4, 5, 6]], [3, 5, 12])

    def test_wrong_dtype(self):
        a = numpy.array([-1, 0, 1])
        b = numpy.array([-2, -1, 0])
        with pytest.raises(ValueError):
            intersection(a, b)


class TestIndexUnion(object):
    def union(self, a, b):
        a = None if a is None else numpy.array(a, dtype=numpy.uint32)
        b = None if b is None else numpy.array(b, dtype=numpy.uint32)
        result = union(a, b)
        return None if result is None else result.tolist()

    def test_union_happy(self):
        a, b = [1, 5, 7, 9, 11], [2, 5, 9, 11, 13]
        assert self.union(a, b) == [1, 2, 5, 7, 9, 11, 13]

    def test_union_empty(self):
        assert self.union([], []) is None
        assert self.union([1, 2, 3], []) == [1, 2, 3]
        assert self.union([], [4, 5, 6]) == [4, 5, 6]
        assert self.union(None, None) is None
        assert self.union([1, 2, 3], None) == [1, 2, 3]
        assert self.union(None, [4, 5, 6]) == [4, 5, 6]

    def test_union_overlap(self):
        # No overlap
        assert self.union([1, 2, 3], [4, 5, 6]) == [1, 2, 3, 4, 5, 6]
        assert self.union([4, 5, 6], [1, 2, 3]) == [1, 2, 3, 4, 5, 6]

        # Overlap by 1
        assert self.union([1, 2, 3], [3, 4, 5]) == [1, 2, 3, 4, 5]
        assert self.union([3, 4, 5], [1, 2, 3]) == [1, 2, 3, 4, 5]

        # Complete overlap
        assert self.union([3, 4, 5], [3, 4, 5]) == [3, 4, 5]

    def test_union_minmax(self):
        M = (2 ** 32) - 1
        assert self.union([0, 100, M], [0, 102, M]) == [0, 100, 102, M]

    def test_multidim_errors(self):
        with pytest.raises(ValueError):
            self.union([[1, 2, 3], [4, 5, 6]], [3, 5, 12])

    def test_wrong_dtype(self):
        a = numpy.array([-1, 0, 1])
        b = numpy.array([-2, -1, 0])
        with pytest.raises(ValueError):
            union(a, b)


class TestIndexDifference(object):
    def difference(self, a, b):
        a = None if a is None else numpy.array(a, dtype=numpy.uint32)
        b = None if b is None else numpy.array(b, dtype=numpy.uint32)
        result = difference(a, b)
        return None if result is None else result.tolist()

    def test_difference_happy(self):
        assert self.difference([1, 5, 7, 9, 11], [2, 5, 9, 11, 13]) == [1, 7]

    def test_difference_empty(self):
        assert self.difference([], []) is None
        assert self.difference([1, 2, 3], []) == [1, 2, 3]
        assert self.difference([], [4, 5, 6]) is None
        assert self.difference(None, None) is None
        assert self.difference([1, 2, 3], None) == [1, 2, 3]
        assert self.difference(None, [4, 5, 6]) is None

    def test_difference_overlap(self):
        # No overlap
        assert self.difference([1, 2, 3], [4, 5, 6]) == [1, 2, 3]
        assert self.difference([4, 5, 6], [1, 2, 3]) == [4, 5, 6]

        # Overlap by 1
        assert self.difference([1, 2, 3], [3, 4, 5]) == [1, 2]
        assert self.difference([3, 4, 5], [1, 2, 3]) == [4, 5]

        # Complete overlap
        assert self.difference([3, 4, 5], [3, 4, 5]) is None

    def test_difference_minmax(self):
        M = (2 ** 32) - 1
        assert self.difference([0, 100, M], [0, 102, M]) == [100]

    def test_multidim_errors(self):
        with pytest.raises(ValueError):
            self.difference([[1, 2, 3], [4, 5, 6]], [3, 5, 12])

    def test_wrong_dtype(self):
        a = numpy.array([-1, 0, 1])
        b = numpy.array([-2, -1, 0])
        with pytest.raises(ValueError):
            difference(a, b)
