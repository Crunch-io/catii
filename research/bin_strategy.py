import os
import timeit

import numpy

from catii.xfuncs import xfunc


def main():
    chart_data = []
    for numrows in (100, 1000, 10000, 100000):
        rowids = numpy.arange(numrows)
        countables_1d = numpy.arange(numrows) % 2
        for numcats in (3, 6, 9, 12, 15):
            coordinates = numpy.empty(numrows, dtype=numpy.uint16)
            for c, rs in enumerate(numpy.array_split(rowids, numcats - 1)):
                coordinates[rs] = c

            for numsubvars in (1, 10, 20, 30, 40, 50, 100):
                countables = numpy.column_stack((countables_1d,) * numsubvars)

                def use_bincount():
                    for i, countables_1d in enumerate(countables.T):
                        output[:, i] = numpy.bincount(
                            coordinates, weights=countables_1d, minlength=numcats
                        )

                def use_bins():
                    for c, rowmask in xfunc.bins(coordinates):
                        output[c] = numpy.sum(countables[rowmask], axis=0)

                output = numpy.zeros((numcats, numsubvars), dtype=int)
                bc_time = min(timeit.repeat(use_bincount, repeat=3, number=1))
                output = numpy.zeros((numcats, numsubvars), dtype=int)
                bin_time = min(timeit.repeat(use_bins, repeat=3, number=1))
                print(
                    "Rows: %d  Cats: %d  Subvars: %d  Bincount time: %s  Bins time: %s  Bincount/bins: %s"
                    % (
                        numrows,
                        numcats,
                        numsubvars,
                        bc_time,
                        bin_time,
                        bc_time / bin_time,
                    )
                )
                chart_data.append(
                    {
                        "strategy": "bincount",
                        "rows": numrows,
                        "categories": numcats,
                        "subvariables": numsubvars,
                        "ms": bc_time * 1000.0,
                    }
                )
                chart_data.append(
                    {
                        "strategy": "bins",
                        "rows": numrows,
                        "categories": numcats,
                        "subvariables": numsubvars,
                        "ms": bin_time * 1000.0,
                    }
                )

    os.chdir(os.path.abspath(os.path.dirname(__file__)))
    import charts

    charts.gen_chart(
        "Bins work",
        [doc for doc in chart_data if doc["strategy"] == "bins"],
        x="subvariables",
        xscale="linear",
        y="ms",
        yscale="linear",
        ylim=(0, None),
        style="rows",
        hue="categories",
    )
    charts.gen_chart(
        "Bin strategy",
        chart_data,
        x="subvariables",
        xscale="linear",
        y="ms",
        yscale="linear",
        ylim=(0, None),
        style="rows",
        hue="strategy",
    )


if __name__ == "__main__":
    main()
