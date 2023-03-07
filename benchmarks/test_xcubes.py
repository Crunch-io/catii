from catii import xcubes, xfuncs

from . import UnitBenchmark


class Test_xcube_1d(UnitBenchmark):
    def _xcube_1d(self, funcname, func, threshold_ms=None):
        C = self.categorical(indexed=False)
        cube = xcubes.xcube([C])
        with self.bench("xcube.1d.%s" % funcname, threshold_ms):
            cube.calculate([func])

    def test_xcube_1d_count(self):
        self._xcube_1d("count", xfuncs.xfunc_count(), 8)

    def test_xcube_1d_valid_count(self):
        A = self.categorical(indexed=False)
        self._xcube_1d("valid_count", xfuncs.xfunc_valid_count(A), 22)


class Test_xcube_1d_x_1d(UnitBenchmark):
    def _xcube_1d_x_1d(self, funcname, func, threshold_ms=None):
        C = self.categorical(indexed=False)
        cube = xcubes.xcube([C, C])
        with self.bench("xcube.1d_x_1d.%s" % funcname, threshold_ms):
            cube.calculate([func])

    def test_xcube_1d_x_1d_count(self):
        self._xcube_1d_x_1d("count", xfuncs.xfunc_count(), 10)

    def test_xcube_1d_x_1d_valid_count(self):
        A = self.categorical(indexed=False)
        self._xcube_1d_x_1d("valid_count", xfuncs.xfunc_valid_count(A), 20)


class Test_xcube_2d(UnitBenchmark):
    def _xcube_2d(self, funcname, func, threshold_ms=None):
        CA = self.categorical(indexed=False, array=True)
        cube = xcubes.xcube([CA])
        with self.bench("xcube.2d.%s" % funcname, threshold_ms):
            cube.calculate([func])

    def test_xcube_2d_count(self):
        self._xcube_2d("count", xfuncs.xfunc_count(), 12000)

    def test_xcube_2d_valid_count(self):
        A = self.categorical(indexed=False)
        self._xcube_2d("valid_count", xfuncs.xfunc_valid_count(A), 35000)
