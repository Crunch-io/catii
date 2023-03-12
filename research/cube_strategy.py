import gc
import os
import timeit

import numpy

from catii import ccube, iindex, xcube
import charts


os.chdir(os.path.abspath(os.path.dirname(__file__)))


class DataGenerator:
    def __init__(self):
        self.ccube_output = {}
        self.xcube_output = {}
        self.weights = {}

    @staticmethod
    def variable(R, C, D, S):
        rowids = numpy.random.choice(R, (R // 100) * D, replace=False)
        index = {}
        for i, rs in enumerate(numpy.array_split(rowids, C - 1)):
            if S is None:
                index[(i,)] = numpy.sort(rs).astype(numpy.uint32)
            else:
                for s in range(S):
                    # Just reuse the same data for all subvars since they're independent.
                    index[(i, s)] = numpy.sort(rs).astype(numpy.uint32)
        return iindex(index, common=C - 1, shape=(R,) if S is None else (R, S))

    def run_one_ccube(self, R, C, D, S, IV, weighted=False, parallel=False):
        key = (R, C, D, S, IV, weighted, parallel)
        if key not in self.ccube_output:
            if weighted:
                if R not in self.weights:
                    self.weights[R] = numpy.random.uniform(0.0, 2.0, R)
                weights = self.weights[R]
            else:
                weights = None
            dims = [self.variable(R, C, D, S) for iv in range(IV)]
            cube = ccube(dims)
            cube.parallel = parallel
            elapsed = min(timeit.repeat(lambda: cube.count(weights), number=1))
            self.ccube_output[key] = elapsed
            print("ccube:", elapsed, key)

            del cube, dims
            gc.collect()
        return self.ccube_output[key]

    def run_one_xcube(self, R, C, D, S, IV, weighted=False, parallel=False):
        key = (R, C, D, S, IV, weighted, parallel)
        if key not in self.xcube_output:
            if weighted:
                if R not in self.weights:
                    self.weights[R] = numpy.random.uniform(0.0, 2.0, R)
                weights = self.weights[R]
            else:
                weights = None
            dims = [self.variable(R, C, D, S).to_array() for iv in range(IV)]
            cube = xcube(dims)
            cube.parallel = parallel
            elapsed = min(timeit.repeat(lambda: cube.count(weights), number=1))
            self.xcube_output[key] = elapsed
            print("xcube:", elapsed, key)

            del cube, dims
            gc.collect()
        return self.xcube_output[key]


gen = DataGenerator()


def gen_2d_rect_cube_time_chart():
    points = []
    for R in (100000, 1000000, 10000000):
        for C in (3, 9, 15):
            for D in (1, 40, 80):
                elapsed = gen.run_one_xcube(R, C, D, S=None, IV=2)
                points.append(
                    {"rows": R, "extent": C, "density": D, "seconds": elapsed,}
                )
    charts.gen_chart(
        "2-D Rect cube time",
        points,
        x="rows",
        xscale="log",
        y="seconds",
        yscale="log",
        style=None,
        hue="density",
    )


def gen_2d_index_cube_time_chart():
    """Generate a chart and linear regression for fill_regions_serial/parallel."""
    # O(∏subvars × ∏extents × N)
    points = []
    for R in reversed((100000, 1000000, 10000000)):
        for C in (3, 9, 15):
            for D in (1, 5, 10, 15, 20):
                elapsed = gen.run_one_ccube(R, C, D, S=None, IV=2)
                points.append(
                    {
                        "rows": R,
                        "extent": C,
                        "density": D,
                        "data points": (R // 100) * D,
                        "seconds": elapsed,
                    }
                )
    charts.gen_chart(
        "2-D Index cube time",
        points,
        x="data points",
        xscale="log",
        y="seconds",
        yscale="log",
        hue="extent",
    )


def gen_2d_index_vs_rect_cube_time_chart():
    points = []
    for R in reversed((1000, 10000, 100000, 1000000, 10000000, 100000000)):
        for C in (3, 9, 15):
            for D in (1, 5, 10, 15, 20):
                elapsed = gen.run_one_ccube(R, C, D, S=None, IV=2)
                points.append(
                    {
                        "algorithm": "Index",
                        "rows": R,
                        "extent": C,
                        "density %": D,
                        "data points": (R // 100) * D,
                        "seconds": elapsed,
                    }
                )
                elapsed = gen.run_one_xcube(R, C, D, S=None, IV=2)
                points.append(
                    {
                        "algorithm": "Rect",
                        "rows": R,
                        "extent": C,
                        "density %": D,
                        "data points": (R // 100) * D,
                        "seconds": elapsed,
                    }
                )
    charts.gen_chart(
        "2-D Index vs Rect cube time",
        points,
        x="rows",
        xscale="log",
        y="seconds",
        yscale="log",
        col="extent",
        hue="algorithm",
        style="density %",
    )


def gen_parallel_index_vs_rect_cube_time_chart():
    points = []
    C = 3
    D = 5
    for R in reversed((1000, 10000, 100000)):  # , 1000000, 10000000, 100000000)):
        for S in (1, 50, 100, 150, 200):
            elapsed = gen.run_one_ccube(R, C, D, S, IV=2, parallel=True)
            points.append(
                {
                    "algorithm": "Index",
                    "rows": R,
                    "subvariables": S,
                    "data points": (R // 100) * D,
                    "seconds": elapsed,
                }
            )
            elapsed = gen.run_one_xcube(R, C, D, S, IV=2, parallel=True)
            points.append(
                {
                    "algorithm": "Rect",
                    "rows": R,
                    "subvariables": S,
                    "data points": (R // 100) * D,
                    "seconds": elapsed,
                }
            )
    charts.gen_chart(
        "2-D Index vs Rect cube time: parallel (extent: 3, density: 5%)",
        points,
        x="subvariables",
        xscale="linear",
        y="seconds",
        yscale="log",
        hue="algorithm",
        style="rows",
    )


if __name__ == "__main__":
    gen_2d_rect_cube_time_chart()
    gen_2d_index_cube_time_chart()
    gen_2d_index_vs_rect_cube_time_chart()
    gen_parallel_index_vs_rect_cube_time_chart()
