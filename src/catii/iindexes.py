import operator
import sys
from collections import defaultdict
from functools import reduce
from itertools import chain, islice

import numpy

from .set_operations import difference, intersection, union

# These are similar to NumPy get/set_printoptions.
_printoptions = {
    # Total number of index elements which trigger summarization
    # rather than full repr (default 10).
    "threshold": 10,
    # Number of index elements in summary at beginning and end (default 3).
    "edgeitems": 3,
}


def fit_dtype(maxval, minval=0):
    """Return a NumPy unsigned-integer dtype wide enough to store the given maxval."""

    # negative side uses more bits
    if maxval < 0 and minval == 0:
        minval = maxval

    if minval < 0:
        if minval < -(2 ** 31):
            dtype = numpy.int64
        elif maxval > 2 ** 31 - 1:
            dtype = numpy.int64
        elif minval < -(2 ** 15):
            dtype = numpy.int32
        elif maxval > 2 ** 15 - 1:
            dtype = numpy.int32
        elif minval < -(2 ** 7):
            dtype = numpy.int16
        elif maxval > 2 ** 7 - 1:
            dtype = numpy.int16
        else:
            dtype = numpy.int8
    else:
        # can use unsigned
        if maxval >= 2 ** 32:
            dtype = numpy.uint64
        elif maxval >= 2 ** 16:
            dtype = numpy.uint32
        elif maxval >= 2 ** 8:
            dtype = numpy.uint16
        else:
            dtype = numpy.uint8
    return numpy.dtype(dtype)


class iindex(dict):
    """An N-dimensional inverted index.

    A dict whose keys are coordinate tuples, and whose corresponding values are
    sorted numpy arrays of row ids.
    """

    common = None
    "The common value. Any rowid not mentioned in self's entries "
    "is assumed to be this value."

    shape = ()
    "A tuple of dimensional extents, such as (3, 4) to mean three rows "
    "and 4 columns. This should match that of an equivalent NumPy array. "
    "May be the empty tuple to signify a zero-dimensional scalar value."

    ROWID_DTYPE = numpy.dtype(numpy.uint32)
    rowid_dtype = ROWID_DTYPE
    "The NumPy dtype of value arrays in self; numpy.uint32 by default."

    def __init__(self, entries, common, shape):
        if type(shape) is not tuple:
            raise TypeError(
                "iindex.shape MUST be of type 'tuple', not %s." % repr(shape)
            )
        self.shape = shape
        self.common = common
        self.rowid_dtype = self.ROWID_DTYPE
        for coords, rowids in entries.items():
            if type(coords) is not tuple:
                raise TypeError("Index coordinates MUST all be tuples.")
            try:
                if rowids.dtype != self.rowid_dtype:
                    raise TypeError(
                        "Index array dtype is %r, but expected %r."
                        % (rowids.dtype, self.rowid_dtype)
                    )
            except AttributeError:
                # Type checks and conversions like this are expensive,
                # and should not be the default, especially in a constructor
                # method which may often simply be re-hydrating values
                # which are known to be well-formed. If you want to avoid
                # this penalty, always pass NumPy arrays.
                if not isinstance(rowids, numpy.ndarray):
                    try:
                        rowids = numpy.asarray(rowids, dtype=self.rowid_dtype)
                    except TypeError:
                        raise TypeError(
                            "Index[%r] could not be converted to an array." % (coords,)
                        )
                    entries[coords] = rowids
                else:
                    raise

        dict.__init__(self, entries)

    def __str__(self):
        if len(self) > _printoptions["threshold"]:
            edges = _printoptions["edgeitems"]
            pairs = ["%r: %r" % (k, v) for k, v in islice(self.items(), edges * 2)]
            entries_part = "{%s, ..., %s}" % (
                ", ".join(pairs[:edges]),
                ", ".join(pairs[edges:]),
            )
        else:
            # If we're below the threshold, sorting should be fast enough.
            pairs = ["%r: %r" % (k, v) for k, v in sorted(self.items())]
            entries_part = "{%s}" % ", ".join(pairs)

        r = "%s(shape=%r, common=%r, entries=%s)" % (
            self.__class__.__name__,
            self.shape,
            self.common,
            entries_part,  # dict as last, because it could get truncated.
        )
        return r

    __repr__ = __str__

    def __eq__(self, other):
        try:
            return (
                self.shape == other.shape
                and self.common == other.common
                and len(self) == len(other)
                and all(
                    len(numpy.setxor1d(rowids, other.get(coords, []))) == 0
                    for coords, rowids in self.items()
                )
            )
        except AttributeError:
            return False

    def validate(self, check_comprehensive_unique=False):
        """Raise ValueError if self is not well-formed.

        This is not called in the constructor or setter methods because:
         1. It can be prohibitively expensive in large datasets.
         2. Constructing instances from known-good data should not pay
            a validation tax.
         3. Building up instances which are temporarily invalid is helpful.
        """
        shape_types = {type(s) for s in self.shape}
        if not shape_types.issubset({int}):
            raise ValueError("Found index shape with wrong types: %s." % (shape_types,))

        for coords, rowids in self.items():
            for c in coords:
                if isinstance(c, numpy.generic):
                    raise ValueError(
                        "Index[%s] contains NumPy coordinate %s." % (coords, c)
                    )

            if coords[0] == self.common:
                raise ValueError(
                    "Index[%s] contains common value %s." % (coords, self.common)
                )

            try:
                if rowids.dtype != self.rowid_dtype:
                    raise ValueError(
                        "Index[%s] is of dtype: %s but should be %s."
                        % (coords, rowids.dtype, self.rowid_dtype)
                    )
            except AttributeError:
                raise ValueError("Index[%s] is not a numpy array." % (coords,))

            sorted_uniq_rowids = numpy.unique(rowids)
            if len(rowids) != len(sorted_uniq_rowids):
                raise ValueError("Index[%s] is not unique." % (coords,))
            if not numpy.array_equal(rowids, sorted_uniq_rowids):
                raise ValueError("Index[%s] is not sorted." % (coords,))

        if check_comprehensive_unique:
            # This check is even MORE expensive, and is therefore extra hard to do.
            rowid_sets = {
                coords: set(rowids.tolist()) for coords, rowids in self.items()
            }
            for coords, rowid_set in rowid_sets.items():
                for other_coords, other_rowid_set in rowid_sets.items():
                    if coords[0] != other_coords[0] and coords[1:] == other_coords[1:]:
                        intersection = rowid_set.intersection(other_rowid_set)
                        if intersection:
                            raise ValueError(
                                "Index[%s] and [%s] contain the same rowids %s."
                                % (coords, other_coords, list(sorted(intersection)))
                            )

    # -------------------------- iindex properties -------------------------- #

    @property
    def abscissae(self):
        """The set of distinct values for the first coordinate."""
        seen = {coords[0] for coords in self}
        if self.size > sum(len(v) for v in self.values()):
            seen.add(self.common)
        return seen

    @property
    def size(self):
        """Number of items in the iindex (including common values)."""
        # This is 13x faster than numpy.prod
        return reduce(operator.mul, self.shape, 1)

    @property
    def sparsity(self):
        """The percentage of items in the iindex which are common."""
        numcells = self.size
        if numcells:
            n_common = numcells - sum(len(v) for v in self.values())
            return (100.0 * n_common) / numcells
        else:
            return 0

    @property
    def nbytes(self):
        """The number of bytes in self."""
        s = sys.getsizeof({})
        s += sum(sys.getsizeof(k) + sys.getsizeof(v) for k, v in self.items())
        s += sys.getsizeof(self.common)
        s += sys.getsizeof(self.shape)
        return s

    @property
    def ndim(self):
        """The number of iindex dimensions."""
        return len(self.shape)

    # -------------------------- iindex conversion -------------------------- #

    def to_array(self, mapping=None, dtype=None):
        """Return a NumPy array of values from self.

        If `mapping` is provided, the returned values will be mapped through it.

        If `dtype` is provided, the returned NumPy array will have that dtype;
        otherwise, an unsigned int dtype will be chosen of sufficient width
        to contain the maximum (possibly mapped) value.
        """
        if not mapping:
            if dtype is None:
                # Always include self.common, in case shape[0] == 0.
                distinct_values = [coords[0] for coords in self] + [self.common]
                vtype = type(distinct_values[0])
                if vtype is int:
                    dtype = fit_dtype(max(distinct_values))
                elif vtype is str:
                    dtype = "<U%d" % max(len(v) for v in distinct_values)
                else:
                    dtype = numpy.object

            output = numpy.full(self.shape, self.common, dtype=dtype)
            if len(self.shape) > 1:
                for coords, rowids in self.items():
                    output[rowids, coords[1]] = coords[0]
            else:
                for coords, rowids in self.items():
                    output[rowids] = coords[0]
        else:
            if dtype is None:
                distinct_values = list(mapping.values())
                vtype = type(distinct_values[0])
                if vtype is int:
                    dtype = fit_dtype(max(distinct_values))
                elif vtype is str:
                    dtype = "<U%d" % max(len(v) for v in distinct_values)
                else:
                    dtype = numpy.object
            output = numpy.full(self.shape, mapping.get(self.common, 0), dtype=dtype)
            if len(self.shape) > 1:
                for (code, col), rowids in self.items():
                    output[rowids, col] = mapping[code]
            else:
                for (code,), rowids in self.items():
                    output[rowids] = mapping[code]

        return output

    @classmethod
    def from_array(cls, values, counts=None, common=None, mapping=None):
        """Return an iindex instance from the given array of values.

        If `counts` is provided, it must be a dict containing each distinct
        input value and its count. If not provided, `bincount` or `unique`
        is called, which can be expensive.

        If `common` is provided, it must be a value of the same type
        as the given values, although it need not be among them.

        If `mapping` is provided, it must be a dict mapping input values
        to output values, and all input values and the provided common value
        (if any) must appear in the mapping.

        The mapping may map multiple inputs to the same output; matching
        rowids will be merged in this case.
        """
        values = numpy.asarray(values)

        if counts is None:
            try:
                if len(values) == 0:
                    values = values.astype(int)  # So bincount doesn't error.
                bcounts = numpy.bincount(values.flat)
                distinct_values = bcounts.nonzero()[0].tolist()
                counts = {i: bcounts[i].item() for i in distinct_values}
            except (ValueError, TypeError):
                try:
                    distinct_values, ucounts = numpy.unique(
                        values.flat, return_counts=True
                    )
                    counts = dict(zip(distinct_values.tolist(), ucounts.tolist()))
                except TypeError:
                    counts = defaultdict(int)
                    for v in values.flat:
                        counts[v] += 1
                    counts = dict(counts)

        if mapping is None:
            final_counts = counts
        else:
            final_counts = defaultdict(int)
            for dv, c in counts.items():
                final_counts[mapping[dv]] += c

        if common is None:
            if len(final_counts):
                common = None
                common_count = None
                for v, c in final_counts.items():
                    if common_count is None or c > common_count:
                        common = v
                        common_count = c
            elif mapping:
                # If a mapping is provided, we can make an initial guess that
                # the common value will be one of the values in that mapping.
                # Pick the lowest one for some stability.
                common = min(mapping.values())
            else:
                # If no information is provided, refuse the temptation to guess.
                raise ValueError("No values or common value provided.")
        else:
            if mapping is not None:
                common = mapping[common]

        rowid_dtype = cls.ROWID_DTYPE

        if values.size == 0 or len(counts) < 5:
            # At a small-enough number of categories, always use numpy.where
            # (the math below loses too much resolution with a tiny numerator anyway).
            use_where = True
            uncommon_ratio = None
        else:
            # numpy.where performs worse with more categories,
            # but better with a higher ratio of uncommon values.
            uncommon_ratio = (
                sum(final_counts.values()) - final_counts[common]
            ) / float(values.size)
            # 100 was determined via benchmarks
            use_where = (len(counts) / uncommon_ratio) < 100

        if use_where:
            entries = {}
            if len(values.shape) > 1:
                for distinct_value in counts:
                    mapped_value = distinct_value
                    if mapping is not None:
                        mapped_value = mapping[distinct_value]
                    if mapped_value == common:
                        continue
                    for colid, col in enumerate(values.T):
                        rowids = numpy.where(col == distinct_value)[0]
                        if len(rowids) > 0:
                            entries[(mapped_value, colid)] = rowids.astype(rowid_dtype)
            else:
                for distinct_value in counts:
                    mapped_value = distinct_value
                    if mapping is not None:
                        mapped_value = mapping[distinct_value]
                    if mapped_value == common:
                        continue
                    rowids = numpy.where(values == distinct_value)[0]
                    if len(rowids) > 0:
                        entries[(mapped_value,)] = rowids.astype(rowid_dtype)
        else:
            # This is sometimes faster than repeated numpy.where().
            entries = defaultdict(list)
            if len(values.shape) > 1:
                for colid, col in enumerate(values.T):
                    for rowid, value in enumerate(col.tolist()):
                        if mapping is not None:
                            value = mapping[value]
                        if value == common:
                            continue
                        entries[(int(value), colid)].append(rowid)
            else:
                for rowid, value in enumerate(values.tolist()):
                    if mapping is not None:
                        value = mapping[value]
                    if value == common:
                        continue
                    entries[(int(value),)].append(rowid)

            entries = {
                coords: numpy.array(rowids, dtype=rowid_dtype)
                for coords, rowids in entries.items()
            }

        return cls(entries, common, values.shape)

    def to_dict(self, force=False):
        """Return a dict of {(coord, ...): [rowids]} pairs from self.

        The returned rowids will be Python lists rather than NumPy arrays.
        This can be especially helpful in tests.
        """
        return {coords: rowids.tolist() for coords, rowids in self.items(force)}

    # ------------------- entry selection and manipulation ------------------- #

    def shift_common(self, new_common=None):
        """Change the most common value in self as needed.

        If `new_common` is None or not provided, the new_common value
        is calculated by counting row ids in self. This is the typical case.
        You should only pass an explicit new_common value when combining
        multiple indexes that may have different common values, in order to
        use the same common value.
        """
        if new_common is None:
            counts = defaultdict(int)
            for coords, rowids in self.items():
                counts[coords[0]] += len(rowids)
            counts[self.common] = self.size - sum(counts.values())
            dsu = max([(v, k) for k, v in counts.items()])
            new_common = dsu[1]

        if new_common != self.common:
            if len(self.shape) > 1:
                mask = numpy.ones(self.shape, dtype=bool)
                for coords, rowids in self.items():
                    mask[rowids, coords[1]] = False
                for col, m in enumerate(mask.T):
                    common_rowids = m.nonzero()[0].astype(self.rowid_dtype)
                    if len(common_rowids):
                        self[(self.common, col)] = common_rowids
            else:
                # We can take a shortcut here
                common_rowids = self.common_rowids()
                if len(common_rowids):
                    self[(self.common,)] = common_rowids

            # Remove any rowids for the new common value
            for coords in list(self.keys()):
                if coords[0] == new_common:
                    del self[coords]

            self.common = new_common

    def common_rowids(self, colindex=None):
        """Return rowids matching the common value (for the given column)."""
        mask = numpy.ones(self.shape[0], dtype=bool)
        if len(self.shape) > 1:
            for coords, rowids in self.items():
                if coords[1] == colindex:
                    mask[rowids] = False
        else:
            for rowids in self.values():
                mask[rowids] = False
        return mask.nonzero()[0].astype(self.rowid_dtype)

    def get(self, key, default=None, force=False):
        """Return rowids for the given coordinate key, or the default.

        The common value is not stored in the index, and when `force` is False
        (the default), it is not returned. Pass `force=True` to return it
        explicitly (by calling common_rowids). This can be expensive and should
        be used sparingly.
        """
        if force and key[0] == self.common:
            rowids = self.common_rowids(*key[1:])
            return default if len(rowids) == 0 else rowids

        return super().get(key, default)

    def set_if(self, key, value, copy=True):
        """If value is None or length 0, pop key, else set self[key] to value.

        If `copy` is True (the default), copy any array value before setting.
        """
        if value is None or len(value) == 0:
            self.pop(key, None)
        else:
            value = numpy.asarray(value)
            self[key] = value.copy() if copy else value

    def items(self, force=False):
        """Return (coords, rowids) pairs from self, plus (common, rowids) if force."""
        if force:
            if len(self.shape) == 1:
                # Calculate common rowids lazily (using a generator comprehension)
                # so if the caller stops iterating before iterating over all
                # items in self, we don't calculate the expensive common rowids.
                commons = (((self.common,), self.common_rowids()) for _ in [1])
            else:
                commons = (
                    ((self.common, colindex), self.common_rowids(colindex))
                    for colindex in range(self.shape[1])
                )
            return chain(self.items(), commons)
        else:
            return super().items()

    # -------------------------- transformed copies -------------------------- #

    def collapsed(self, precedence, mapping=None):
        """Return a new iindex, collapsing axis 1 in precedence order.

         ____M____
         A   B   C     M.collapsed([1, 0, -1])
        --  --  --     -----------------------
        -1   0  -1                           0
         1   0  -1                           1
         0   0   0                           0
         0   1  -1                           1
        -1  -1  -1                          -1

        The returned index will be of the same row length, but without columns.
        Each output row will contain the first value from the given "precedence"
        argument which is present in at least one column. That is, given
        precedence [1, 0, -1], a row will obtain a 1 if any of the input
        index's columns contains a 1; if none do, the row will obtain
        a 0 if any of the columns contains a 0; if none do, the row
        will obtain a -1.

        Not all existing values need to be included in the "precedence" argument;
        any rows which only contain unmentioned ids will obtain the last value
        in the given precedence.
        """
        if len(self.shape) < 2:
            raise TypeError(
                "Cannot collapse: no column axis present "
                "in shape %s." % repr(self.shape)
            )

        if mapping is None:
            new_common = self.common
        else:
            new_common = mapping.get(self.common, self.common)

        numrows, numcols = self.shape[:2]
        if not numrows:
            return self.__class__({}, new_common, (0,))

        # Gather all rowids for each (possibly mapped) initial coordinate.
        gathered = {}
        for coords, rowids in self.items():
            new_coord = coords[0]
            if mapping is not None:
                new_coord = mapping.get(new_coord, new_coord)

            if new_coord != new_common:
                v = gathered.get(new_coord)
                if v is not None:
                    v.append(rowids)
                else:
                    gathered[new_coord] = [rowids]

        # Iterate through the gathered rowids in reverse precedence order,
        # overwriting the output as we go.
        # This takes some RAM but only O(rows), not subvars etc.
        dtype = fit_dtype(max(precedence))
        default = precedence[-1]
        output = numpy.full(numrows, default, dtype=dtype)
        common_has_been_written = True
        if default != new_common:
            # We filled the output with the lowest-precedence coord.
            # If that's NOT the common value, then we need to keep track
            # of which rows have explicitly obtained an uncommon value.
            common_has_been_written = False
            common_count = numpy.full(numrows, numcols, dtype=fit_dtype(numcols))
            for rowids in gathered.get(default, []):
                common_count[rowids] -= 1
        for coord in reversed(precedence[:-1]):
            if coord == new_common:
                # Rows which already have ALL values at a lower precedence
                # stay that way; any others get the common value for now,
                # (but may be overwritten with higher precedence later).
                output[common_count != 0] = coord
                common_has_been_written = True
            else:
                for rowids in gathered.get(coord, []):
                    output[rowids] = coord
                    if not common_has_been_written:
                        # This simple flag can save a lot of runtime, only
                        # counting values "to the right of" the common value.
                        common_count[rowids] -= 1

        # from_array will determine the new best common value for us.
        return self.__class__.from_array(output)

    def copy(self):
        """Return a copy of self."""
        return iindex(
            dict((coords, rowids.copy()) for coords, rowids in self.items()),
            self.common,
            self.shape,
        )

    def filtered(self, mask, new_length):
        """Return a copy of self, masked by the given boolean array row mask.

        Index  Dense       Map         Mask    Index  Dense        Map
          0      7     7: [0, 2, 4]     T        0      7      7: [0, 3]
          1      8     8: [1, 3, 5]     T        1      8      8: [1, 2]
          2      7                  --- F --->   ?
          3      8                      T        2      8
          4      7                      T        3      7
          5      8                      F        ?
        """
        new_rowids = numpy.empty(len(mask), dtype=self.rowid_dtype)
        new_rowids[mask] = numpy.arange(new_length, dtype=self.rowid_dtype)

        new_entries = {}
        for coords, rowids in self.items():
            m = mask[rowids]
            if numpy.any(m):
                filtered_rowids = rowids[m]
                new_entries[coords] = new_rowids[filtered_rowids]

        new_shape = (new_length,) + self.shape[1:]
        new_index = self.__class__(new_entries, self.common, new_shape)
        new_index.shift_common()

        return new_index

    def sliced(self, *orders):
        """Return a copy of self, with data for the given orders only.

        Each argument addresses a higher dimension. That is, if one argument
        is passed, the second axis is sliced; if two are passed, then
        the second and third dimensions are sliced. Pass `None` to skip
        an axis without slicing it.

        Each argument may be an integer, a list of integers, or None.
        If a list, then those slices are included in the output in that order.
        If a single integer, only that slice is included, and that axis is
        collapsed in the output. If None, that axis is included unchanged.
        """
        if not orders:
            return self

        if len(orders) > self.ndim - 1:
            raise TypeError(
                "Cannot slice %d axes with shape %r." % (len(orders), self.shape)
            )

        new_shape = [self.shape[0]]
        for i, order in enumerate(orders, 1):
            if order is None:
                new_shape.append(self.shape[i])
            elif isinstance(order, int):
                pass
            else:
                new_shape.append(len(order))
        new_shape = tuple(new_shape)

        new_entries = {}
        for coords, rowids in self.items():
            keep = True
            new_coords = [coords[0]]
            for axis, order in enumerate(orders, 1):
                coord = coords[axis]
                if order is None:
                    # Keep all slices for this axis.
                    new_coords.append(coord)
                elif isinstance(order, int):
                    if coord == order:
                        # Keep this single slice for this axis
                        # (but drop the axis).
                        pass
                    else:
                        keep = False
                        break
                else:
                    if coord in order:
                        new_coords.append(order.index(coord))
                    else:
                        keep = False
                        break

            if keep:
                new_entries[tuple(new_coords)] = rowids

        return iindex(new_entries, self.common, new_shape)

    def reindexed(self, mapping, copy=True, shift=True, assume_unique=False):
        """Return a new iindex, whose entries contain mapped values.

        The mapping may map multiple inputs to the same output; matching
        rowids will be merged in this case. If you can guarantee that
        there will be no duplicates (no merged coords for the same rowid),
        you can save some execution time by passing `assume_unique=True`.

        If `copy` is True (the default), the rowids in the returned entries
        are materialized copies of self. If False, they are shared.
        Use False only when self is already a temporary copy, or will not
        be persisted.
        """
        new_common = mapping.get(self.common, self.common)

        merged = False

        new_entries = {}
        for coords, rowids in self.items():
            if not hasattr(coords, "__iter__"):
                coords = (coords,)

            new_coord = mapping.get(coords[0])
            if new_coord == new_common:
                # More than one coord maps to the new common coord.
                # Skip, but flag so that common is shifted below.
                merged = True
                continue
            if new_coord is not None:
                coords = (new_coord,) + coords[1:]

            v = new_entries.get(coords)
            if v is not None:
                v.append(rowids)
                merged = True
            else:
                new_entries[coords] = [rowids]

        for coords, rowid_lists in list(new_entries.items()):
            if len(rowid_lists) > 1:
                # When the number of categories grows large (eg 100),
                # this is much faster than heapq, and slightly faster than
                # sortednp.kway_merge (or even custom specializations of it).
                rowids = numpy.concatenate(rowid_lists)
                rowids.sort()
                if not assume_unique:
                    mask = numpy.empty(rowids.shape, dtype=bool)
                    mask[:1] = True
                    mask[1:] = rowids[1:] != rowids[:-1]
                    rowids = rowids[mask]
            else:
                rowids = rowid_lists[0]
                if copy:
                    rowids = rowids.copy()
            new_entries[coords] = rowids

        new_index = self.__class__(new_entries, new_common, self.shape)
        if shift and merged:
            # One or more sets of rowids was combined.
            # See if the common value needs shifting.
            new_index.shift_common()

        return new_index

    def slices1d(self, base_coords=()):
        """Yield recursive (coords, 1-D slice) pairs of self.

        This is useful for aggregations (like counting) over higher dimensions.
        Slices are iterated over in REVERSE axis order--that is, the last
        coordinate is the outermost loop--on the theory that increasing axes
        describe increasingly higher dimensions (arrays of arrays).
        The returned coordinate tuples are in the original order.

        For example, an iindex with shape (1000, 2, 3) would yield 1-D slices
        in this order:
            (0, 0): idx.slice([0], [0]),
            (1, 0): idx.slice([1], [0]),
            (2, 0): idx.slice([2], [0]),
            (0, 1): idx.slice([0], [1]),
            (1, 1): idx.slice([1], [1]),
            (2, 1): idx.slice([2], [1]),
        """
        if len(self.shape) > 1:
            # Slice...
            buckets = [{} for coord in range(self.shape[-1])]
            for coords, rowids in self.items():
                buckets[coords[-1]][coords[:-1]] = rowids

            # ...and recurse
            subshape = self.shape[:-1]
            for coord, subentries in enumerate(buckets):
                for s in iindex(subentries, self.common, subshape).slices1d(
                    (coord,) + base_coords
                ):
                    yield s
        else:
            yield base_coords, self

    # -------------------------- combining iindexes -------------------------- #

    def append(self, other):
        """Vertically stack the given `other` index as new rows below self.

        Any rowids in `other` will be incremented by self.shape[0].

        The `other` iindex must be of the same type (its entries must mean
        the same things as self), but it need not have the same common value.
        """
        if not isinstance(other, iindex):
            raise TypeError(
                "Can't append object of type %r to iindex." % (type(other),)
            )

        old_numrows = self.shape[0]
        new_numrows = old_numrows + other.shape[0]
        shift = self.rowid_dtype.type(old_numrows)
        dtype = self.ROWID_DTYPE

        if len(self.shape) > 1:
            for coords, new_rowids in other.items():
                if coords[0] != self.common:
                    shifted_rowids = new_rowids.astype(dtype) + shift
                    rowids = self.get(coords)
                    if rowids is None:
                        self[coords] = shifted_rowids
                    else:
                        self[coords] = numpy.append(rowids, shifted_rowids)
            if other.common != self.common:
                # Rowids for other.common were not appended above. Do so now.
                for col in range(self.shape[1]):
                    shifted_rowids = other.common_rowids(col).astype(dtype) + shift
                    rowids = self.get((other.common, col))
                    if rowids is None:
                        self[(other.common, col)] = shifted_rowids
                    else:
                        self[(other.common, col)] = numpy.append(rowids, shifted_rowids)
        else:
            for coords, new_rowids in other.items():
                if coords[0] != self.common:
                    shifted_rowids = new_rowids.astype(dtype) + shift
                    rowids = self.get(coords)
                    if rowids is None:
                        self[coords] = shifted_rowids
                    else:
                        self[coords] = numpy.append(rowids, shifted_rowids)
            if other.common != self.common:
                # Rowids for other.common were not appended above. Do so now.
                shifted_rowids = other.common_rowids().astype(dtype) + shift
                rowids = self.get((other.common,))
                if rowids is None:
                    self[(other.common,)] = shifted_rowids
                else:
                    self[(other.common,)] = numpy.append(rowids, shifted_rowids)

        self.shape = (new_numrows,) + self.shape[1:]
        self.shift_common()

    def update(self, entries):
        """Update self with the given entries, which may contain the common value.

        This method does NOT shift the common value even if the density changes.

        The given entries are not a complete iindex, and therefore
        have no shape or common value. They are not required to represent all
        cells in the rectangular array; indeed the whole point is to not do so.
        """
        # Each new value is replacing an old value at a particular rowid.
        # Do this in two passes, where we first *remove* the old association
        # by deleting the target rowids from self...
        # TODO: benchmark doing this with set difference of each key with
        # all rows for same axes; would probably save memory but explode time.
        other_cell_mask = numpy.zeros(self.shape, dtype=bool)
        for coords, new_rowids in entries.items():
            other_cell_mask[(new_rowids,) + coords[1:]] = True

        to_delete = []
        for coords, rowids in self.items():
            matches = other_cell_mask[(rowids,) + coords[1:]]
            if numpy.any(matches):
                if numpy.all(matches):
                    to_delete.append(coords)
                else:
                    self[coords] = rowids[~matches]
        for coords in to_delete:
            del self[coords]

        # ...and then insert new rowids.
        self.union_update({k: v for k, v in entries.items() if k[0] != self.common})

    def union_update(self, other):
        """Update self, adding elements from other.

        If `other` is an iindex, its common value is not unioned--only its
        explicit entries. You may need to shift_common() on self or other
        before updating.
        """
        for coords, rowids in other.items():
            rowids = numpy.asarray(rowids, dtype=self.rowid_dtype)
            # Make a copy of other[coords] if coords not in self.
            self.set_if(coords, union(self.get(coords), rowids, copy_right=True))

    def intersection_update(self, other):
        """Update self, keeping only elements found in it and other.

        If `other` is an iindex, its common value is not intersected--only its
        explicit entries. You may need to shift_common() on self or other
        before updating.
        """
        for coords in list(self.keys()):
            if coords not in other:
                del self[coords]

        for coords, rowids in other.items():
            rowids = numpy.asarray(rowids, dtype=self.rowid_dtype)
            self.set_if(coords, intersection(self.get(coords), rowids))

    def difference_update(self, other):
        """Update self, removing elements found in others.

        If `other` is an iindex, its common value is not differenced--only its
        explicit entries. You may need to shift_common() on self or other
        before updating.
        """
        for coords, rowids in other.items():
            rowids = numpy.asarray(rowids, dtype=self.rowid_dtype)
            # There's no need to copy self[coords] when updating self.
            self.set_if(coords, difference(self.get(coords), rowids, copy=False))
