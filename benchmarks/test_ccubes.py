from catii import ccubes, ffuncs

from . import UnitBenchmark


class Test_ccube_1d(UnitBenchmark):
    def _ccube_1d(self, funcname, func, threshold_ms=None):
        C = self.categorical()
        cube = ccubes.ccube([C])
        with self.bench("ccube.1d.%s" % funcname, threshold_ms):
            cube.calculate([func])

    def test_ccube_1d_count(self):
        self._ccube_1d("count", ffuncs.ffunc_count(), 1)

    def test_ccube_1d_valid_count(self):
        A = self.categorical(indexed=False)
        self._ccube_1d("valid_count", ffuncs.ffunc_valid_count(A), 8)


class Test_ccube_1d_x_1d(UnitBenchmark):
    def _ccube_1d_x_1d(self, funcname, func, threshold_ms=None):
        C = self.categorical()
        cube = ccubes.ccube([C, C])
        with self.bench("ccube.1d_x_1d.%s" % funcname, threshold_ms):
            cube.calculate([func])

    def test_ccube_1d_x_1d_count(self):
        self._ccube_1d_x_1d("count", ffuncs.ffunc_count(), 25)

    def test_ccube_1d_x_1d_valid_count(self):
        A = self.categorical(indexed=False)
        self._ccube_1d_x_1d("valid_count", ffuncs.ffunc_valid_count(A), 40)


class Test_ccube_2d(UnitBenchmark):
    def _ccube_2d(self, funcname, func, threshold_ms=None):
        CA = self.categorical(array=True)
        cube = ccubes.ccube([CA])
        with self.bench("ccube.2d.%s" % funcname, threshold_ms):
            cube.calculate([func])

    def test_ccube_2d_count(self):
        self._ccube_2d("count", ffuncs.ffunc_count(), 90)

    def test_ccube_2d_valid_count(self):
        A = self.categorical(indexed=False)
        self._ccube_2d("valid_count", ffuncs.ffunc_valid_count(A), 4500)
