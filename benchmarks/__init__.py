"""Benchmarking for catii.

Our primary goal here is to produce and maintain a product which is fast
and reliable. This requires:
    1. A system which monitors for significant changes in performance.
    2. Tools to easily explain poor performance in our highly complex product.
    3. Tools to easily develop a product with improved performance.

To meet those needs, we here defined a set of benchmarks which are run via
py.test like the rest of our test suite.

This package, particularly cubes, can be complex, with lots of nested processing.
Please do your best to write "unit benchmarks" that test small pieces. For example,
it can be tempting to look at a slow query in production and drop the whole
expression into a benchmark. However, that doesn't scale; you should find the
worst offender, and add a benchmark just for it. This will be much
easier to improve, demonstrate, and then monitor for regression.

By the same token, you should not add separate benchmarks which only vary
by the number of variables, rows, axes, categories, etc. You should pick
the worst combination and only test that. The base classes here provide
helpers for exactly that, defaulting to sensible numbers for these parameters.
At least, they're sensible for single operations--in combination, they can be
too large, and individual benchmarks should override them when warranted.
For example, when testing xcube.product, a default of 4 dimensions is good,
but you should set `self.params.rows = 10` since a) that won't affect
the test result, but b) leaving it at the default of a million rows
would waste several minutes in setup time and possibly lock up your computer.

You *should* add separate benchmarks for separate code paths. However,
you should take care to separate concerns--if an operation takes 10 times
as long due to a "cold load", consider adding a separate unit benchmark
for the load step, rather than a "cold" and "hot" benchmark for each
affected operation.
"""

import gc
import itertools
import time
import unittest
from contextlib import contextmanager

import numpy
import pytest

from catii import iindex


class BenchmarkParameters(dict):
    """A helper for passing around parameters in benchmark runs.

    Benchmarks need to set parameters, such as for variable size and shape,
    and also emit those in the reported results, so that multiple runs
    with different parameters can be distinguished. Rather than pass
    the same handful of arguments through every test method and helper
    function, we set them here.

    There is some extra magic here concerning default parameters. We want to
    report all parameters that are actually in use by the benchmark, but not
    those that are not. For example, if a function takes a single variable
    as the left-hand input, we want a benchmark to report the number
    of rows in that variable. We want most benchmarks to default to 1M rows,
    so that we don't have to repeat "rows=1000000" everywhere. Other benchmarks,
    like cubes, might default to `variables=4`; however, the single-argument
    function can only take 1 variable as input, so we do not want its benchmark
    to report `variables=4` or even `variables=1` when that has no meaning.
    We therefore record params only when accessed explicitly, either by the
    test method itself, or by calling a helper method to construct variables etc.
    """

    _defaults = {
        "rows": 1000000,
        "categories": 100,
        "axes": (1000,),
        "density": 0.5,
    }

    def __getattr__(self, key):
        if key in self:
            return self[key]
        elif key in self._defaults:
            self[key] = self._defaults[key]
            return self[key]

        raise AttributeError(
            "'BenchmarkParameters' object has no attribute '%s'" % (key,)
        )

    def __setattr__(self, key, value):
        self[key] = value

    @contextmanager
    def __call__(self, **fields):
        old_params = dict(self)
        self.update(fields)
        try:
            yield
        finally:
            self.clear()
            self.update(old_params)


class UnitBenchmark(unittest.TestCase):

    maxDiff = None

    def setUp(self):
        super().setUp()
        self.params = BenchmarkParameters()

    def tearDown(self):
        self.params = None
        super().tearDown()

    @contextmanager
    def bench(self, name, threshold_ms=None):
        """Wrap the given context in a benchmark.

        If `threshold_ms` is not None, it must be the number of
        milliseconds that the context is expected to take (as an upper bound).
        If the context takes more time than this to complete, the test xfails.

        This should be a one-way ratchet: never increased, but occasionally
        decreased as effort is devoted to user happiness.
        """
        gc.collect()
        diff_s = None

        start = time.perf_counter()
        yield
        diff_s = time.perf_counter() - start

        if self.params:
            paramstr = "\t".join(
                ["%s=%s" % (k, v) for k, v in sorted(self.params.items())]
            )
        else:
            paramstr = ""
        print("\n%10.6f" % diff_s, name, paramstr)

        if threshold_ms is not None:
            if (diff_s * 1000) > threshold_ms:
                msg = "Benchmark %r time %.3fms > threshold %.3fms" % (
                    name,
                    diff_s * 1000,
                    threshold_ms,
                )
                pytest.xfail(msg)

    def categorical(self, indexed=True, array=False):
        # Don't muck about with distributions; just fill the first rows.
        rowids = numpy.arange(
            int(self.params.rows * self.params.density), dtype=iindex.ROWID_DTYPE,
        )
        index = {
            (c + 1,): rs
            for c, rs in enumerate(
                numpy.array_split(rowids, self.params.categories - 1)
            )
        }

        shape = (self.params.rows,)
        if array:
            # Re-use the rowids for each additional axis since they're independent.
            index = {
                k + coords: v
                for k, v in index.items()
                for coords in itertools.product(*[range(e) for e in self.params.axes])
            }
            shape = shape + self.params.axes

        var = iindex(index, common=0, shape=shape)
        if not indexed:
            var = var.to_array(dtype=numpy.uint16)

        return var

    def numeric(self):
        return (numpy.arange(self.params.rows) % 100) / 100.0
