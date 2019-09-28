from glob import glob
import operator
import os

import humanize
import numpy
from pyspark import RDD
from scipy.sparse import csr_matrix

from sourced.ml.models import BOW, OrderedDocumentFrequencies
from sourced.ml.transformers import Indexer, Uast2BagFeatures
from sourced.ml.transformers.transformer import Transformer


class BOWWriter(Transformer):
    DEFAULT_CHUNK_SIZE = 2 * 1000 * 1000 * 1000

    def __init__(self, document_indexer: Indexer, df: OrderedDocumentFrequencies,
                 filename: str, series: str, chunk_size: int = DEFAULT_CHUNK_SIZE, **kwargs):
        super().__init__(**kwargs)
        self.document_indexer = document_indexer
        self.df = df
        self.filename = filename
	self.series = series
        self.chunk_size = chunk_size

    def __getstate__(self):
        state = super().__getstate__()
        del state["document_indexer"]
        del state["token_indexer"]
        del state["df"]
        return state

    def __call__(self, head: RDD):
        c = Uast2BagFeatures.Columns
        self._log.info("Estimating the average bag size...")
        avglen = head \
            .map(lambda x: (x[c.document], 1)) \
            .reduceByKey(operator.add) \
            .map(lambda x: x[1])
        if self.explained:
            self._log.info("toDebugString():\n%s", avglen.toDebugString().decode())
        avglen = avglen.mean()
        self._log.info("Result: %.0f", avglen)
        avgdocnamelen = numpy.mean([len(v) for v in self.document_indexer.value_to_index])
        nparts = int(numpy.ceil(
            len(self.document_indexer) * (avglen * (4 + 4) + avgdocnamelen) / self.chunk_size))
        self._log.info("Estimated number of partitions: %d", nparts)
        doc_index_to_name = {
            index: name for name, index in self.document_indexer.value_to_index.items()}
        tokens = self.df.tokens()
        it = head \
            .map(lambda x: (x[c.document], (x[c.token], x[c.value]))) \
            .groupByKey() \
            .repartition(nparts) \
            .glom() \
            .toLocalIterator()
        if self.explained:
            self._log.info("toDebugString():\n%s", it.toDebugString().decode())
        ndocs = 0
        self._log.info("Writing files to %s", self.filename)
        for i, part in enumerate(it):
            docs = [doc_index_to_name[p[0]] for p in part]
            if not len(docs):
                self._log.info("Batch %d is empty, skipping.", i + 1)
                continue
            size = sum(len(p[1]) for p in part)
            data = numpy.zeros(size, dtype=numpy.float32)
            indices = numpy.zeros(size, dtype=numpy.int32)
            indptr = numpy.zeros(len(docs) + 1, dtype=numpy.int32)
            pos = 0
            for pi, (_, bag) in enumerate(part):
                for tok, val in sorted(bag):
                    indices[pos] = tok
                    data[pos] = val
                    pos += 1
                indptr[pi + 1] = indptr[pi] + len(bag)
            assert pos == size
            matrix = csr_matrix((data, indices, indptr), shape=(len(docs), len(tokens)))
            filename = self.get_bow_file_name(self.filename, i)
            BOW() \
                .construct(docs, tokens, matrix) \
                .save(filename, series=self.series, deps=(self.df,))
            self._log.info("%d -> %s with %d documents, %d nnz (%s)",
                           i + 1, filename, len(docs), size,
                           humanize.naturalsize(os.path.getsize(filename)))
            ndocs += len(docs)
        self._log.info("Final number of documents: %d", ndocs)

    def get_bow_file_name(self, base: str, index: int):
        parent, full_name = os.path.split(base)
        name, ext = os.path.splitext(full_name)
        return os.path.join(parent, "%s_%03d%s" % (name, index + 1, ext))


class BOWLoader:
    def __init__(self, glob_path):
        self.files = glob(glob_path)

    def __len__(self):
        return len(self.files)

    def __bool__(self):
        return bool(self.files)

    def __iter__(self):
        return (BOW().load(path) for path in self.files)
