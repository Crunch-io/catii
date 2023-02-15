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

### Indexes versus arrays

Although the iindex class can represent the same information as a NumPy array,
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
contiguous integers starting from 0. This is required when using ccubes
(below), so storing data as integers saves precious query time. Any names
or other attributes should be separate metadata.

## The contingency cube (ccube)

Analyses generally consist of summary statistics of variables (iindexes
or arrays), quite often contingent on, or "grouped by", other iindexes.
For example, you might want to know the number of people who are affiliated
with a particular political party (say, 0=None, 1=Rep, 2=Dem), but grouped
by education level to see if there is any significant correlation between
the two variables. With catii, you calculate this using a ccube:

```#python
>>> from catii import ccube, iindex
>>> party = iindex({(1,): [0, 2, 5], (2,): [4]}, common=0, shape=(8,))
>>> educ = iindex({(0,): [2, 5, 4], (2,): [4]}, common=1, shape=(8,))
>>> ccube([educ, party]).count()
array([[1, 2, 0],
       [3, 1, 0],
       [0, 0, 1]])
```

The returned array is a 2-dimensional "hypercube": a contingency table
(or "crosstab") representing the frequency distribution of these two variables
for our sample. You're probably familiar seeing it in a more tabular form:

```
         party
          0    1    2
e        --   --   --
d    0 |  1    2    0
u    1 |  3    1    0
c    2 |  0    0    1
```

We passed our two variables as dimensions to the ccube, and asked for a count
of their interaction. Since we provided two 1-D iindexes, the output was 2-D.
If either of our input iindexes had additional axes, the output would be three
or more dimensions.

The dimensions must all correspond row-wise; that is, when we say "the educ
variable has value 0 in row 5" and "the party variable has value 1 in row 5",
we mean that "row 5" in both refers to the same observation in our data.
We aggregate over the rows, so they are never a dimension in our output.

When the dimension iindexes include additional axes, they are assumed to be
independent, and the ccube iterates over their Cartesian product. For each
combination of higher axes, it forms a subcube of 1-D slices of each dimension,
and stacks their output. For example, if instead of "party" we had a 2-D
"genre" variable for recording which music genres (say, 0=classical, 1=pop",
and 2=alternative) people like or dislike (0=missing, 1=like, 2=dislike),
we would see an additional dimension in our cube output. The like/dislike
axis would be distinct values and therefore the first coordinate in our
iindex tuples. The genre axis would be placed in the second coordinates.

```#python
>>> genre = iindex.from_array([
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [2, 1, 1],
    [1, 0, 0],
    [2, 2, 1]
    ])
>>> genre
iindex(shape=(6, 3), common=0, entries={
    (1, 0): array([4], dtype=uint32),
    (1, 1): array([2, 3], dtype=uint32),
    (1, 2): array([1, 3, 5], dtype=uint32),
    (2, 0): array([3, 5], dtype=uint32),
    (2, 1): array([5], dtype=uint32)
    })
>>> ccube([genre]).count()
array([[3, 1, 2],  # classical
       [3, 2, 1],  # pop
       [3, 3, 0]]) # alternative
```

The additional axes are always moved to be outermost (in reverse precedence),
so the result above iterates over the genre axis first, and then more tightly
over the missing/like/dislike values.

### Frequency functions (ffuncs)

In addition to counting, catii provides other aggregate functions to use with
cubes: sum, mean, and valid count. These all take at least one additional "fact
variable" as an argument; that is, the data to sum, or average, or count valid
values within. These must also correspond row-wise to the cube dimensions.

Each of these operate just like count, and return a cube of results.

### Weights

All of the ffuncs take an optional "weights" argument. If given, it must also
correspond row-wise to the dimensions and any fact variables. The function then
weights the data by these values. For example:

```#python
>>> from catii import ccube, iindex
>>> import numpy
>>> party = iindex({(1,): [0, 2, 5], (2,): [4]}, common=0, shape=(8,))
>>> ccube([party]).count()
array([4, 3, 1])
>>> weights = numpy.arange(10) / 10.0
>>> weights
array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
>>> ccube([party]).count(weights)
array([3.4, 0.7, 0.4])
```

### Missing values

Fact variables and weight variables may take one of two forms:
    * a single NumPy array, where missing values are represented by NaN or NaT
    * a pair of NumPy arrays, where the first contains values and the second
      is a "validity" array of booleans: True meaning "valid" and False meaning
      "missing". Where False, the corresponding values in the first array
      are ignored.

To some extent, which format you choose depends on your application and how
your data is already represented. Note, however, that NumPy arrays of `int`
or `str` have no standard way to represent missing values. Rather than nominate
sentinel values for these and similar types, you may pass a separate "validity"
array of booleans, and you might therefore consider doing so for all dtypes.
Note this is slightly faster, as well.

Functions provided here all have a `reduce` method which returns cube output
as NumPy arrays; these outputs may also have a missing value in any cell that
1) had no rows in the inputs with cube dimensions corresponding to it, or
2) had rows but corresponding fact values or weights were missing values ("all"
if `ignore_missing` else "any").

This is somewhat divergent from standard NumPy which, for example, defaults
numpy.sum([]) to 0.0. However, consumers of catii often need to distinguish
a sum or mean of [0, 0] from one of []. You are, of course, free to take the
output and set arr[arr.isnan()] = 0 if you desire.

Each ffunc has an optional `ignore_missing` arg. If False (the default), then
any missing values (values=NaN or validity=False) are propagated so that
outputs also have a missing value in any cell that had a missing value in
one of the rows in the fact variable or weight that contributed to that cell.
If `ignore_missing` is True, such input rows are ignored and do not contribute
to the output, much like NumPy's `nanmean` or R's `na.rm = TRUE`. Note this
is also faster and uses less memory.

The `reduce` methods herein all default to the "single NumPy array" format,
with NaN values indicating missingness. Pass e.g. `return_missing_as=(0, False)`
to return a 2-tuple of (values, validity) arrays instead. Functions here will
replace NaN values in the `values` array with 0 in that case. If you prefer
`sum([])` to return 0, for example, without a second "validity" array,
pass `return_missing_as=0`.

### Combined cube calculation

Often, when finding summaries like a weighted count, we also want an unweighted
count, or we want to show means but with the "base" counts. We could simply
form one cube and call each shortcut method:

```#python
>>> c = ccube([educ, party])
>>> c.mean(arr)
>>> c.count()
```

However, that has to form the interaction of the dimensions twice. If our educ
and party variables have millions of rows, or are very dense, or have hundreds
of categories, or additional axes, or if we cross additional variables, this
step can quickly multiply in execution time. You can save time by using
ccube.calculate instead and pass a list of ffuncs:

```#python
>>> from catii import ffuncs
>>> c = ccube([educ, party])
>>> means, counts = c.calculate([ffuncs.ffunc_mean(arr), ffuncs.ffunc_count()])
```

This iterates over our educ and party dimensions once, and passes
each distinct combination of coordinates to each ffunc.
