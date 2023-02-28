import mmap
import struct

import numpy

from .iindexes import fit_dtype


class IndxIO(object):
    """A reader/writer for the INDX format (*unsigned integer values only*)

    All integers are unsigned, little-endian.

    8-byte magic+version "INDX0001"
    buffer size           e0 00 00 00 00 00 00 00     (224)
    index dimensions      02
    index length          04 00 00 00
    index word size       01/02/04/08               + index word size governs bytes in:
    index common value    00                      <--
    index                 01 00 01 01 02 00 02 01 <--
    rowid word size       01/02/04/08               + rowid word size governs bytes in:
    rowid lengths         02 02 01 01             <--
    rowids                03 05                   <--
                          01 04
                          02
                          05
    """

    INDEXED_MAGIC = b"INDX"
    VERSION = b"0001"

    @staticmethod
    def load(f):
        """Read file (in INDX format) and return entries, common, rowid dtype."""
        if f.read(4) != IndxIO.INDEXED_MAGIC:
            raise RuntimeError("Unexpected header")

        version = f.read(4)
        if version != IndxIO.VERSION:
            raise RuntimeError("Unexpected indexed format %s" % version)

        buffer_size = struct.unpack("<Q", f.read(8))[0]

        offset = 16
        buffer_length = offset + buffer_size
        buf = mmap.mmap(
            f.fileno(), buffer_length, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ
        )
        entries = {}

        # Read index dimensions
        index_dimensions = struct.unpack_from("<B", buf, offset=offset)[0]
        offset += 1

        # Read index length
        index_length = struct.unpack_from("<L", buf, offset=offset)[0]
        offset += 4

        # Read index word size
        index_word_size = struct.unpack_from("<B", buf, offset=offset)[0]
        ind_format_string = IndxIO.format(index_word_size)
        offset += 1

        # Read common value
        common = struct.unpack_from(ind_format_string, buf, offset=offset)[0]
        offset += index_word_size

        # Read index
        index = numpy.ndarray(
            shape=(index_length, index_dimensions),
            buffer=buf,
            dtype=IndxIO.dtype(index_word_size),
            offset=offset,
        )
        offset += index.nbytes
        # TODO: at least 50% of the time is this tolist() call.
        # Investigate how to remove it, but understand why it's there:
        # we don't want to put ourselves in a place where we have to call
        # [x.item() if isinstance(x, ndarray) else x] and similar things when
        # comparing Python category/subvariable indexes to numpy ones,
        # or serializing to JSON, etc. It's subtle and should not be
        # changed without lots of research.
        all_coords = [tuple(row) for row in index.tolist()]
        if all_coords and any(type(c) is not int for c in all_coords[0]):
            # In Python 2, ndarray.tolist() will return Python `long` values
            # when the dtype is uint32/64. We want `int`.
            # This works for values up to 2 ** 63 - 1:
            # >>> int(long(2 ** 63 - 1))
            # 9223372036854775807
            # ...but not any larger values. Let's hope we never have that many.
            # >>> int(long(2 ** 63))
            # 9223372036854775808L
            # TODO: remove this when Python 2 is no longer supported.
            all_coords = [tuple([int(c) for c in row]) for row in all_coords]

        # Read rowids word size
        word_size = struct.unpack_from("<B", buf, offset=offset)[0]
        rowid_dtype = IndxIO.dtype(word_size)
        offset += 1

        # Read rowid lengths
        lengths = numpy.ndarray(
            shape=(len(all_coords),), buffer=buf, dtype=rowid_dtype, offset=offset
        )
        offset += len(lengths) * word_size

        # Read rowids. It's  about 15% faster to make one ndarray and slice it
        # 1M times compared to 1M ndarrays over the same buffer.
        rowid_lists = numpy.ndarray(
            shape=int((buffer_length - offset) / rowid_dtype.itemsize),
            buffer=buf,
            dtype=rowid_dtype,
            offset=offset,
        )
        ptr = 0
        for length, coords in zip(lengths, all_coords):
            rowids = rowid_lists[ptr : ptr + length]
            ptr += length
            # For now, force uint32 everywhere.
            # Later we can grow/shrink this if needed.
            if rowids.dtype != numpy.uint32:
                # This will break the mmap of any array, and is only here
                # to help migrate variables on alpha which are not 32-bit
                rowids = rowids.astype(numpy.uint32)
            entries[coords] = rowids

        return entries, common, rowid_dtype

    @staticmethod
    def save(f, entries, common, dtype):
        """Write the given entries to the open file in INDX format."""
        if len(entries) > 2 ** 32:
            raise ValueError("Cannot save indexed variable: too many coordinates.")

        # We iterate over entries keys twice, let's make a copy of it.
        list_index = list(entries.keys())
        index = numpy.array(list_index)
        lengths = numpy.array(
            [len(entries[coords]) for coords in list_index], dtype=dtype
        )

        index_dtype = fit_dtype(
            max(numpy.max(index), common) if len(index) != 0 else common
        )
        index_word_size = index_dtype.itemsize

        # numpy will usually pick int64, this will get called often
        if index.dtype != index_dtype:
            index = index.astype(index_dtype, copy=False)

        buffer_size = (
            1  # index dimensions
            + 4  # index length
            + 1  # index word size
            + index_word_size  # index common value
            + index.nbytes  # index
            + 1  # rowid word size
            + len(lengths) * dtype.itemsize  # rowid lengths
            + sum(lengths) * dtype.itemsize  # rowids
        )

        f.write(IndxIO.INDEXED_MAGIC)
        f.write(IndxIO.VERSION)

        # Write buffer size
        f.write(struct.pack("<Q", buffer_size))

        ind_format_string = IndxIO.format(index_word_size)

        # Write index dimensions
        f.write(struct.pack("<B", 0 if index.ndim == 1 else index.shape[1]))

        # Write index length
        f.write(struct.pack("<L", len(index)))

        # Write index word size
        f.write(struct.pack("<B", index_word_size))

        # Write common value
        f.write(struct.pack(ind_format_string, common))

        # Write index
        index.tofile(f)

        # Write rowid word size
        f.write(struct.pack("<B", dtype.itemsize))

        # Write rowid lengths
        lengths.tofile(f)

        # Write rowids
        for i in list_index:
            arr = entries[i]
            if arr.dtype != dtype:
                raise RuntimeError(
                    (
                        "Illegal indexed data: "
                        "array is of dtype %s but should be of dtype %s."
                    )
                    % (arr.dtype, dtype)
                )
            arr.tofile(f)

        if f.tell() != 16 + buffer_size:
            raise RuntimeError("Wrote illegal index format: wrong length.")

    @staticmethod
    def format(size):
        """return string for struct.pack format."""

        if isinstance(size, numpy.dtype):
            size = size.itemsize

        if size == 8:
            return "<Q"
        elif size == 4:
            return "<L"
        elif size == 2:
            return "<H"
        return "<B"

    @staticmethod
    def dtype(itemsize):
        """return numpy.dtype of unsigned int of desired itemsize."""
        if itemsize == 8:
            return numpy.dtype(numpy.uint64)
        elif itemsize == 4:
            return numpy.dtype(numpy.uint32)
        elif itemsize == 2:
            return numpy.dtype(numpy.uint16)
        return numpy.dtype(numpy.uint8)
