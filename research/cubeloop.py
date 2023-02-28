"""Code to choose the fastest custom-code-in-recursive-loop strategy.

One of the core goals of catii is low cube completion time,
especially as the number of dimensions, distinct values, and rows
grows large. Since we iterate over all of these in nested loops,
it's important to optimize the looping itself just as much as the
work performed by each iteration.

Python is a great language for orchestrating work, but it has a performance
weakness: calling functions is slow. Most time-critical operations on data
points should be done on structured arrays in C, like NumPy does. We never
want to call a Python function per data point! But even per dimension or
distinct value can add up.

In Python, yielding from a generator pays most of this function-call overhead,
as it switches execution frames from the callee to the caller and back.
Simple testing shows `yield` pays about 60% of the overhead of the equivalent
`walk` approach where one passes a callback. But this is misleading;
when the iteration is *recursive*, and one has to `yield from` each
inner loop, the cost exceeds that of passing in a callback--up to 60% more
in the example below.

Conveniently, this makes composition simpler. Rather than repeat a `for` loop
in each aggregation function, we can instead focus each function on a single
coordinate, and even perform multiple aggregations in a single loop.
"""

import os
import timeit


def yielder(dims):
    remaining_dims = dims[1:]
    if remaining_dims:
        for c in dims[0]:
            yield from yielder(remaining_dims)
    else:
        for c in dims[0]:
            yield c


def yielding():
    for c in yielder(dims):
        a = c


def walk(dims, func):
    remaining_dims = dims[1:]
    if remaining_dims:
        for c in dims[0]:
            walk(remaining_dims, func)
    else:
        for c in dims[0]:
            func(c)


def walking():
    def f(c):
        a = c

    walk(dims, f)


def main():
    data = []
    for d in range(1, 5):
        for c in (1, 5, 10, 15):
            global dims
            dims = [list(range(c)) for x in range(d)]
            y_time = min(timeit.repeat(yielding, number=10000))
            w_time = min(timeit.repeat(walking, number=10000))
            print(
                "Dims: %d  Cats: %d  Yield time: %s  Walk time: %s  Yield/walk: %s"
                % (d, c, y_time, w_time, y_time / w_time)
            )
            data.append(
                {
                    "strategy": "yield",
                    "dimensions": d,
                    "categories": c,
                    "ms": y_time * 1000.0,
                }
            )
            data.append(
                {
                    "strategy": "walk",
                    "dimensions": d,
                    "categories": c,
                    "ms": w_time * 1000.0,
                }
            )

    os.chdir(os.path.abspath(os.path.dirname(__file__)))
    import charts

    charts.gen_chart(
        "Cube looping strategy",
        data,
        x="categories",
        xscale="linear",
        y="ms",
        yscale="symlog",
        ylim=(0, None),
        style="dimensions",
        hue="strategy",
    )


if __name__ == "__main__":
    main()
