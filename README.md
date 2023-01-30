# catii :cat::eyes:

A library for N-dimensional categorical data.

## The inverted index (iindex)

Multidimensional data values may be represented in a variety of ways.
A very common format is the "rectangular" or "tabular" or "wide" format,
an N-dimensional array where each observation is a "row" and each cell
value is a "category id" or some other value. This can then be extended
to two or more dimensions by adding axes as "columns" and higher dimensions.

The same information can be stored as an "inverted index" as we do here
by storing tuples of coordinates, including the "category id" as the
first coordinate; "columns" or higher dimensions are then additional
ordinates in the tuple. Then, for each set of coordinates, we associate
it with an array of the "row ids" where that combination occurs.

The inverted index can be much smaller than the equivalent array
by omitting the most common category from the index. Any row whose
row id is not present in the index is assumed to have the common
value. In many fields, the larger the dataset, the more sparse it is.

For example, the rectangular array::

```#python
a = [1, 0, 4, 0, 1, 1, 4, 1]
```

...becomes an iindex with shape (8,), a "common" value of 1, and two entries::


```#python
>>> from catii import iindex
>>> iindex.from_array(a)
iindex(shape=(8,), common=1, entries={(0,): array([1, 3], dtype=uint32), (4,): array([2, 6], dtype=uint32)})
```

The rectangular array::
```#python
b = [
    [2, 2, 2],
    [2, 0, 2],
    [2, 2, 4],
    [2, 0, 2],
    [2, 2, 2],
    [2, 2, 4],
]
```

...becomes an iindex with shape (6, 3), a "common" value of 2, and the 2-D entries::


```#python
>>> iindex.from_array(b)
iindex(shape=(6, 3), common=2, entries={(0, 1): array([1, 3], dtype=uint32), (4, 2): array([2, 5], dtype=uint32)})
>>> _.to_array()
array([[2, 2, 2],
       [2, 0, 2],
       [2, 2, 4],
       [2, 0, 2],
       [2, 2, 2],
       [2, 2, 4]], dtype=uint8)
```

It is generally assumed that iindex.common is the most common value, but it
is possible to construct an iindex instance where this is not true.
Call shift_common() to normalize this. Consequently, two indexes
could represent the same exact data values but with different common values
if one or both have not been normalized; these will not be equal using `==`.

## Indexes versus arrays

Although the iindex class can represent the same data as a NumPy array,
it purposefully does NOT implement the NumPY API, because mixing arrays and
indexes often results in crudely transforming the sparse index into a dense
array, obliterating all of the benefits of the index. Iterating over all of
the values in an index should not be easy. Instead, the index implements
methods which are essential to 1) statistical analysis, and 2) altering
and interacting with other indexes.

Indexes generally perform as well or better than arrays at these tasks--up to
about 75% density for one dimension, and 40% for two or more dimensions.
If you have large, dense arrays, or arrays which contain thousands of
distinct values, you should consider a hybrid index/array approach or
just use arrays or some other approach.

Each entry's array of rowids is assumed to be sorted, which allows several
optimizations. However, in the interest of speed, the index class itself
does NOT automatically validate this. You may call index.validate() to check
this and many other invariants, but keep in mind that this is generally
slow, and should be avoided in production code once proven.

The first coordinate is always the "mutually exclusive" category; additional
coordinates are not exclusive. For example, row id 0 should not appear
at both (1, 7) and (2, 7), but it could appear at both (1, 7) and (1, 8).
Imagine a variable "do you like this sandwich?" where "1" means "yes"
and "2" means "no" for the first coordinate, and "7" means "cheesesteak"
while "8" means "BLT" for the second coordinate. It's unlikely you need
to represent someone choosing both "yes" and "no" for "cheesesteak",
but quite common to represent someone choosing "yes" for both sandwiches.

Unlike most array implementations, the machine type of the distinct values is
not a focus--they may be any hashable Python type. In practice, especially
when mixing indexes with arrays, it is generally best to treat coordinates
as exactly that: points in a linear space of distinct values, usually
contiguous integers starting from 0. Any names or other attributes
should be separate metadata.
