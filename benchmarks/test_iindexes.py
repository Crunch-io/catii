import numpy

from catii import iindex

from . import UnitBenchmark


class Test_iindex_reindexed(UnitBenchmark):
    def test_iindex_reindexed_unique(self):
        C = self.categorical()
        with self.bench("iindex.reindexed", 2):
            C.reindexed()

    def test_iindex_reindexed_merge(self):
        C = self.categorical()
        mapping = {k: k % 4 for k in range(self.params.categories)}
        with self.bench("iindex.reindexed.merge", 8):
            C.reindexed(mapping)

        self.params.axes = (100,)
        C = self.categorical(array=True)
        mapping = {k: k % 4 for k in range(self.params.categories)}
        with self.bench("iindex.reindexed.merge.array", 500):
            C.reindexed(mapping)

    def test_iindex_reindexed_merge_interleaved(self):
        # In real data each category's rows are spread over all rows,
        # not one block each as in self.categorical().
        values = numpy.arange(self.params.rows) % self.params.categories
        C = iindex.from_array(values)
        mapping = {k: k % 4 for k in range(self.params.categories)}
        with self.bench("iindex.reindexed.merge.interleaved", None):
            C.reindexed(mapping)
