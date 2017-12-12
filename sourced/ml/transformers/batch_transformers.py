import os
from collections import namedtuple
from glob import glob

import numpy
import parquet
from pyspark.sql.types import Row
from scipy.sparse import csr_matrix

from sourced.ml.models import OrderedDocumentFrequencies
from sourced.ml.transformers import Transformer
from sourced.ml.utils import PickleableLogger


class BagsBatcher(Transformer):
    DEFAULT_CHUNK_SIZE = 1.5 * 1000 * 1000 * 1000
    BLOCKS = True

    def __init__(self, extractors, chunk_size=DEFAULT_CHUNK_SIZE, **kwargs):
        super().__init__(**kwargs)
        self.extractors = extractors
        self.model = None
        self.chunk_size = chunk_size

    def __getstate__(self):
        state = super().__getstate__()
        del state["extractors"]
        del state["model"]
        if self.model is not None:
            state["keys"] = self.model.order
        return state

    def __call__(self, processed):
        lengths = processed.values().map(len)
        if self.explained:
            self._log.info("toDebugString():\n%s", lengths.toDebugString().decode())
        avglen = lengths.mean()
        ndocs = self.extractors[0].ndocs
        self._log.info("Average bag length: %.1f", avglen)
        self._log.info("Number of documents: %d", ndocs)
        self.model = OrderedDocumentFrequencies().construct(
            self.extractors[0].ndocs, [e.docfreq for e in self.extractors])
        self._log.info("Vocabulary size: %d", len(self.model))
        chunklen = int(self.chunk_size / (2 * 4 * avglen))
        nparts = ndocs // chunklen + 1
        chunklen = int(ndocs / nparts * (2 * 4 * avglen))
        self._log.info("chunk %d\tparts %d", chunklen, nparts)
        return processed.mapValues(self.bag2row).repartition(nparts)

    def bag2row(self, bag):
        data = numpy.zeros(len(bag), dtype=numpy.float32)
        indices = numpy.zeros(len(bag), dtype=numpy.int32)
        # self.keys emerges after __getstate__
        for i, (k, v) in enumerate(sorted((self.keys[k], v) for k, v in bag.items())):
            data[i] = v
            indices[i] = k
        return data, indices


class BagsBatchSaver(Transformer):
    def __init__(self, path, batcher, **kwargs):
        super().__init__(**kwargs)
        self.path = path
        self.batcher = batcher
        self.vocabulary_size = 0

    def __getstate__(self):
        state = super().__getstate__()
        del state["batcher"]
        return state

    def __call__(self, head):
        self._log.info("Writing to %s", self.path)
        self.vocabulary_size = len(self.batcher.model)
        rows = head \
            .mapPartitions(self.concatenate) \
            .map(lambda x: Row(**x))
        if self.explained:
            self._log.info("toDebugString():\n%s", rows.toDebugString().decode())
        rows.toDF().write.parquet(self.path)

    def concatenate(self, part):
        data = []
        indices = []
        indptr = [0]
        keys = []
        for k, (d, i) in part:
            keys.append(k)
            data.append(d)
            indices.append(i)
            indptr.append(indptr[-1] + d.shape[0])
        data = numpy.concatenate(data).astype(dtype=numpy.float32)
        indices = numpy.concatenate(indices).astype(dtype=numpy.int32)
        indptr = numpy.array(indptr, dtype=numpy.int64)
        return [{"keys": bytearray("\0".join(keys).encode()),
                 "data": bytearray(data.data),
                 "indices": bytearray(indices.data),
                 "indptr": bytearray(indptr.data),
                 "rows": indptr.shape[0] - 1,
                 "cols": self.vocabulary_size}]


BagsBatch = namedtuple("BagsBatch", ("keys", "matrix"))


class BagsBatchParquetLoader(PickleableLogger):
    class BagsBatchParquetLoaderIterator(PickleableLogger):
        def __init__(self, files, **kwargs):
            super().__init__(**kwargs)
            self._files = files
            self._pos = 0

        def _get_log_name(self):
            return BagsBatchParquetLoader.__name__

        def __next__(self):
            if self._pos < len(self._files):
                f = self._files[self._pos]
                self._pos += 1
                try:
                    with open(f, "rb") as fin:
                        rows = list(parquet.DictReader(fin))
                        if len(rows) != 1:
                            raise ValueError("%s contains more than one row" % f)
                        row = rows[0]
                        keys = row["keys"].decode().split("\0")
                        matrix = csr_matrix((numpy.frombuffer(row["data"], numpy.float32),
                                             numpy.frombuffer(row["indices"], numpy.int32),
                                             numpy.frombuffer(row["indptr"], numpy.int64)),
                                            shape=(row["rows"], row["cols"]))
                        return BagsBatch(keys=keys, matrix=matrix)
                except Exception as e:
                    self._log.error("Loading %s: %s: %s", f, type(e).__name__, e)
            raise StopIteration()

    def __init__(self, path, **kwargs):
        super().__init__(**kwargs)
        self._files = glob(os.path.join(path, "*.parquet"))

    def _get_log_name(self):
        return type(self).__name__

    def __iter__(self):
        return self.BagsBatchParquetLoaderIterator(self._files, log_level=self._log.level)
