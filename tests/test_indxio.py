import tempfile

import numpy
import pytest

from catii import indxio


class TestIndxIO:
    def test_save_indexed_errors_with_too_many_coordinates(self):
        """Make sure we error is we try to save too many coordinates.
        The current implementation saves length of index as a
        packed integer that shouldn't exceed 4 bytes.
        """

        class FakeDict:
            def __len__(self):
                return 2 ** 33

        with tempfile.TemporaryFile() as f:
            with pytest.raises(ValueError, match="too many coordinates"):
                indxio.IndxIO.save(f, FakeDict(), 0, numpy.dtype("<u8"))

    def test_indxio_vector_roundtrip(self):
        with tempfile.TemporaryFile() as f:
            saved_dtype = numpy.dtype(numpy.uint32)
            saved_entries = {
                (0,): numpy.array([1, 2, 5], dtype=saved_dtype),
                (1,): numpy.array([10, 20, 60], dtype=saved_dtype),
                # This will fit in uint32, but a .tolist() from that
                # will return coordinates that are Python 2 `long`, not `int`.
                (2 ** 20,): numpy.array([100, 200, 600], dtype=saved_dtype),
            }
            saved_common = 3
            indxio.IndxIO.save(f, saved_entries, saved_common, saved_dtype)

            f.seek(0)

            (loaded_entries, loaded_common, loaded_rowid_dtype,) = indxio.IndxIO.load(f)

            assert loaded_common == saved_common
            assert loaded_rowid_dtype == saved_dtype
            assert {k: v.tolist() for k, v in loaded_entries.items()} == {
                k: v.tolist() for k, v in saved_entries.items()
            }
            assert set(type(k[0]) for k in loaded_entries) == {int}

    def test_indxio_array_roundtrip(self):
        with tempfile.TemporaryFile() as f:
            saved_dtype = numpy.dtype(numpy.uint32)
            saved_entries = {
                (0, 0): numpy.array([1, 2, 5], dtype=saved_dtype),
                (0, 2): numpy.array([1, 2, 6], dtype=saved_dtype),
                (1, 2): numpy.array([1, 2, 6], dtype=saved_dtype),
            }
            saved_common = 3
            indxio.IndxIO.save(f, saved_entries, saved_common, saved_dtype)

            f.seek(0)

            (loaded_entries, loaded_common, loaded_rowid_dtype,) = indxio.IndxIO.load(f)

            assert loaded_common == saved_common
            assert loaded_rowid_dtype == saved_dtype
            assert {k: v.tolist() for k, v in loaded_entries.items()} == {
                k: v.tolist() for k, v in saved_entries.items()
            }

    def test_indxio_empty_entries_roundtrip(self):
        with tempfile.TemporaryFile() as f:
            saved_dtype = numpy.dtype(numpy.uint32)
            saved_entries = {}
            saved_common = 3
            indxio.IndxIO.save(f, saved_entries, saved_common, saved_dtype)

            f.seek(0)

            (loaded_entries, loaded_common, loaded_rowid_dtype,) = indxio.IndxIO.load(f)

            assert loaded_common == saved_common
            assert loaded_rowid_dtype == saved_dtype
            assert {k: v.tolist() for k, v in loaded_entries.items()} == {
                k: v.tolist() for k, v in saved_entries.items()
            }

    def test_indxio_3d_roundtrip(self):
        with tempfile.TemporaryFile() as f:
            saved_dtype = numpy.dtype(numpy.uint32)
            saved_entries = {
                (0, 0, 0): numpy.array([1, 2, 5], dtype=saved_dtype),
                (0, 2, 3): numpy.array([1, 2, 6], dtype=saved_dtype),
                (1, 2, 4096): numpy.array([1, 2, 6], dtype=saved_dtype),
            }
            saved_common = 3
            indxio.IndxIO.save(f, saved_entries, saved_common, saved_dtype)

            f.seek(0)

            (loaded_entries, loaded_common, loaded_rowid_dtype,) = indxio.IndxIO.load(f)

            assert loaded_common == saved_common
            assert loaded_rowid_dtype == saved_dtype
            assert {k: v.tolist() for k, v in loaded_entries.items()} == {
                k: v.tolist() for k, v in saved_entries.items()
            }
