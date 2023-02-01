"""Cython speedups for set operations on iindex rowids.

Because iindex rowids are required to be sorted arrays, set operations like
union can often be faster with a "mergesort" approach: iterate through
both arrays, taking the smaller value each time, to produce sorted output.
Set intersection and difference can take a similar approach, but taking
equal or unequal values, with an even greater speedup.
"""

# cython: profile=False, language_level=3
import numpy
cimport cython
cimport numpy

ctypedef numpy.uint32_t uint32


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def set_intersect_merge_np(const uint32[:] left_array, const uint32[:] right_array):
    """Return the set intersection of the given ordered arrays."""

    # Note that, because these ptr/len vars are `int`, they'll choke if one
    # of the input arrays has len >= 2 ** 31 (about 2 billion).
    # That SHOULD be an anomaly, but we may have to make this polymorphic
    # on the length of the input arrays. It's not like we can handle more
    # than 2 ** 32 with uint32 anyway, it's just that gap between 31 and 32.
    cdef int left_ptr, right_ptr
    cdef uint32 left, right
    cdef int left_len = left_array.shape[0]
    cdef int right_len = right_array.shape[0]

    # Form a result array which we will fill with the set intersection results.
    # The output cannot be any longer than the smaller of the two input arrays,
    # so we form an output array of that length and truncate it on exit to the
    # filled length.
    cdef int max_result_len = min(left_len, right_len)
    result = numpy.empty(max_result_len, dtype=numpy.uint32)
    cdef uint32[:] result_view = result
    cdef int result_len = 0

    with nogil:
        if left_len > 0 and right_len > 0:
            left_ptr = 0
            right_ptr = 0
            left = left_array[left_ptr]
            right = right_array[right_ptr]

            if (left > right_array[right_len - 1]) or (right > left_array[left_len - 1]):
                # The two arrays do not overlap at all.
                pass
            else:
                while 1:
                    if left > right:
                        # Right value not present in left array.
                        right_ptr += 1
                        if right_ptr >= right_len:
                            break
                        right = right_array[right_ptr]
                    elif right > left:
                        # Left value not present in right array.
                        left_ptr += 1
                        if left_ptr >= left_len:
                            break
                        left = left_array[left_ptr]
                    else:
                        result_view[result_len] = left
                        result_len += 1

                        left_ptr += 1
                        right_ptr += 1
                        if left_ptr >= left_len:
                            break
                        if right_ptr >= right_len:
                            break
                        left = left_array[left_ptr]
                        right = right_array[right_ptr]

    return result[:result_len]


def intersection(left_array, right_array):
    """Return the set intersection of the given ordered arrays, or None.

    If either value is None, or the result is zero length, None is returned.
    """
    if left_array is None or right_array is None:
        return None
    else:
        result = set_intersect_merge_np(left_array, right_array)
        if len(result):
            return result


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def set_union_merge_np(const uint32[:] left_array, const uint32[:] right_array):
    """Return the (ordered) set union of the given ordered arrays."""

    # Note that, because these ptr/len vars are `int`, they'll choke if one
    # of the input arrays has len >= 2 ** 31 (about 2 billion).
    # That SHOULD be an anomaly, but we may have to make this polymorphic
    # on the length of the input arrays. It's not like we can handle more
    # than 2 ** 32 with uint32 anyway, it's just that gap between 31 and 32.
    cdef int left_ptr, right_ptr
    cdef uint32 left, right
    cdef int left_len = left_array.shape[0]
    cdef int right_len = right_array.shape[0]

    # Form a result array which we will fill with the set intersection results.
    # The output cannot be longer than the concatenation of the two input arrays,
    # so we form an output array of that length and truncate it on exit to the
    # filled length.
    cdef int max_result_len = left_len + right_len
    result = numpy.empty(max_result_len, dtype=numpy.uint32)
    cdef uint32[:] result_view = result
    cdef int result_len = 0

    if left_len == 0:
        return numpy.asarray(right_array)
    elif right_len == 0:
        return numpy.asarray(left_array)

    left_ptr = 0
    right_ptr = 0
    left = left_array[left_ptr]
    right = right_array[right_ptr]

    if left > right_array[right_len - 1]:
        # The two arrays do not overlap at all.
        return numpy.concatenate((right_array, left_array))
    elif right > left_array[left_len - 1]:
        # The two arrays do not overlap at all.
        return numpy.concatenate((left_array, right_array))

    with nogil:
        while 1:
            if left > right:
                # Right value not present in left array.
                result_view[result_len] = right
                result_len += 1
                right_ptr += 1
                if right_ptr >= right_len:
                    break
                right = right_array[right_ptr]
            elif right > left:
                # Left value not present in right array.
                result_view[result_len] = left
                result_len += 1
                left_ptr += 1
                if left_ptr >= left_len:
                    break
                left = left_array[left_ptr]
            else:
                result_view[result_len] = left
                result_len += 1

                left_ptr += 1
                right_ptr += 1
                if left_ptr >= left_len:
                    break
                if right_ptr >= right_len:
                    break
                left = left_array[left_ptr]
                right = right_array[right_ptr]
        while left_ptr < left_len:
            result_view[result_len] = left_array[left_ptr]
            result_len += 1
            left_ptr += 1
        while right_ptr < right_len:
            result_view[result_len] = right_array[right_ptr]
            result_len += 1
            right_ptr += 1

    return result[:result_len]


def union(left_array, right_array, copy_left=False, copy_right=False):
    """Return the set union of the given ordered arrays, or None.

    If both values are None, or the result is zero length, None is returned.
    """
    if left_array is None:
        if right_array is None:
            return None
        else:
            result = right_array.copy() if copy_right else right_array
    else:
        if right_array is None:
            result = left_array.copy() if copy_left else left_array
        else:
            result = set_union_merge_np(left_array, right_array)

    if len(result):
        return result


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def set_union_merge_many(list arrays):
    """Return the (ordered) set union of the given ordered arrays."""

    # Note that, because these ptr/len vars are `int`, they'll choke if one
    # of the input arrays has len >= 2 ** 31 (about 2 billion).
    # That SHOULD be an anomaly, but we may have to make this polymorphic
    # on the length of the input arrays. It's not like we can handle more
    # than 2 ** 32 with uint32 anyway, it's just that gap between 31 and 32.
    cdef list value_arrays = [arr for arr in arrays if len(arr)]
    cdef long num_arrays = len(value_arrays)
    varr = numpy.concatenate(value_arrays)
    cdef uint32[:] values = varr
    larr = numpy.array([arr.shape[0] for arr in value_arrays], dtype=int)
    cdef long[:] lengths = larr
    parr = numpy.concatenate([[0], lengths[:len(larr) - 1]])
    cdef long[:] pointers = parr
    limarr = parr + larr
    cdef long[:] limits = limarr
    cdef uint32 limit_value = max([arr[len(arr) - 1] for arr in value_arrays]) + 1

    # Form a result array which we will fill with the set intersection results.
    # The output cannot be longer than the concatenation of the input arrays,
    # so we form an output array of that length and truncate it on exit to the
    # filled length.
    result = numpy.empty(len(values), dtype=numpy.uint32)
    cdef uint32[:] result_view = result
    cdef int result_len = 0
    cdef long arrnum, min_arrnum, ptr = 0
    cdef uint32 value, min_value = 0

    with nogil:
        while 1:
            # Find the minimum value and its array number.
            min_value = limit_value
            min_arrnum = -1
            for arrnum in range(num_arrays):
                ptr = pointers[arrnum]
                if ptr >= limits[arrnum]:
                    continue
                value = values[ptr]
                if value < min_value:
                    min_value = value
                    min_arrnum = arrnum

            if min_value == limit_value:
                # All arrays have been exhausted.
                break

            result_view[result_len] = min_value
            result_len += 1

            pointers[min_arrnum] += 1

    return result[:result_len]


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def set_difference_merge_np(const uint32[:] left_array, const uint32[:] right_array):
    """Return the elements in left that are not in right (all ordered arrays)."""

    # Note that, because these ptr/len vars are `int`, they'll choke if one
    # of the input arrays has len >= 2 ** 31 (about 2 billion).
    # That SHOULD be an anomaly, but we may have to make this polymorphic
    # on the length of the input arrays. It's not like we can handle more
    # than 2 ** 32 with uint32 anyway, it's just that gap between 31 and 32.
    cdef int left_ptr, right_ptr
    cdef uint32 left, right
    cdef int left_len = left_array.shape[0]
    cdef int right_len = right_array.shape[0]

    # Form a result array which we will fill with the set intersection results.
    # The output cannot be any longer than the left array,
    # so we form an output array of that length and truncate it on exit to the
    # filled length.
    result = numpy.empty(left_len, dtype=numpy.uint32)
    cdef uint32[:] result_view = result
    cdef int result_len = 0

    if left_len == 0:
        pass
    elif right_len == 0:
        result[:left_len] = left_array
        result_len = left_len
    else:
        left_ptr = 0
        right_ptr = 0
        left = left_array[left_ptr]
        right = right_array[right_ptr]

        if (left > right_array[right_len - 1]) or (right > left_array[left_len - 1]):
            # The two arrays do not overlap at all, so return left.
            result[:left_len] = left_array
            result_len = left_len
        else:
            with nogil:
                while 1:
                    if left > right:
                        # Right value not present in left array.
                        right_ptr += 1
                        if right_ptr >= right_len:
                            break
                        right = right_array[right_ptr]
                    elif right > left:
                        # Left value not present in right array.
                        result_view[result_len] = left
                        result_len += 1

                        left_ptr += 1
                        if left_ptr >= left_len:
                            break
                        left = left_array[left_ptr]
                    else:
                        left_ptr += 1
                        right_ptr += 1
                        if left_ptr >= left_len:
                            break
                        if right_ptr >= right_len:
                            break
                        left = left_array[left_ptr]
                        right = right_array[right_ptr]
                while left_ptr < left_len:
                    result_view[result_len] = left_array[left_ptr]
                    result_len += 1
                    left_ptr += 1

    return result[:result_len]


def difference(left_array, right_array, copy=False):
    """Return the set difference (left - right) of the given ordered arrays, or None.

    If the left array is None, or the result is zero length, None is returned.
    """
    if left_array is None:
        return None
    if right_array is None:
        result = left_array.copy() if copy else left_array
    else:
        result = set_difference_merge_np(left_array, right_array)
    if len(result):
        return result
