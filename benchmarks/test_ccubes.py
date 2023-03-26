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

    def test_ccube_1d_count_wt(self):
        W = self.numeric()
        self._ccube_1d("count.wt", ffuncs.ffunc_count(weights=W), 8)

    def test_ccube_1d_valid_count(self):
        A = self.categorical(indexed=False)
        self._ccube_1d("valid_count", ffuncs.ffunc_valid_count(A), 10)

    def test_ccube_1d_valid_count_wt(self):
        A = self.categorical(indexed=False)
        W = self.numeric()
        self._ccube_1d("valid_count.wt", ffuncs.ffunc_valid_count(A, weights=W), 12)

    def test_ccube_1d_sum(self):
        N = self.numeric()
        self._ccube_1d("sum", ffuncs.ffunc_sum(N), 12)

    def test_ccube_1d_sum_wt(self):
        N = self.numeric()
        W = self.numeric()
        self._ccube_1d("sum.wt", ffuncs.ffunc_sum(N, weights=W), 12)

    def test_ccube_1d_mean(self):
        N = self.numeric()
        self._ccube_1d("mean", ffuncs.ffunc_mean(N), 16)

    def test_ccube_1d_mean_wt(self):
        N = self.numeric()
        W = self.numeric()
        self._ccube_1d("mean.wt", ffuncs.ffunc_mean(N, weights=W), 15)


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
        self._ccube_1d_x_1d("valid_count", ffuncs.ffunc_valid_count(A), 45)


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
        self._ccube_2d("valid_count", ffuncs.ffunc_valid_count(A), 5000)


class Test_ccube_multiple_ffuncs(UnitBenchmark):
    def test_ccube_multiple_ffuncs(self):
        C = self.categorical()
        cube = ccubes.ccube([C, C])
        N = self.numeric()
        fs = [
            ffuncs.ffunc_count(),
            ffuncs.ffunc_valid_count(N),
            ffuncs.ffunc_sum(N),
            ffuncs.ffunc_mean(N),
        ]
        with self.bench("ccube.multiple_ffuncs"):
            cube.calculate(fs)
