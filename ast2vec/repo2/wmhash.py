import copy
import itertools

import numpy

from ast2vec.pickleable_logger import PickleableLogger
from ast2vec.repo2.base import Repo2Base
from ast2vec.uast_ids_to_bag import UastIds2Bag


__extractors__ = {}


def register_extractor(cls):
    if not issubclass(cls, BagsExtractor):
        raise TypeError("%s is not an instance of %s" % (cls.__name__, BagsExtractor.__name__))
    __extractors__[cls.NAME] = cls
    return cls


class BagsExtractor:
    DEFAULT_DOCFREQ_THRESHOLD = 5

    def __init__(self, docfreq_threshold=None):
        if docfreq_threshold is None:
            docfreq_threshold = self.DEFAULT_DOCFREQ_THRESHOLD
        self.docfreq_threshold = docfreq_threshold
        self.docfreq = {}
        self._ndocs = 0

    @property
    def docfreq_threhold(self):
        return self._docfreq_threshold

    @docfreq_threhold.setter
    def docfreq_threshold(self, value):
        if not isinstance(value, int):
            raise TypeError("docfreq_threshold must be an integer, got %s" % type(value))
        if value < 1:
            raise ValueError("docfreq_threshold must be >= 1, got %d" % value)
        self._docfreq_threshold = value

    @property
    def ndocs(self):
        return self._ndocs

    @ndocs.setter
    def ndocs(self, value):
        if not isinstance(value, int):
            raise TypeError("ndocs must be an integer, got %s" % type(value))
        if value < 1:
            raise ValueError("ndocs must be >= 1, got %d" % value)
        self._ndocs = value

    def extract(self, uast):
        raise NotImplementedError()

    def inspect(self, uast):
        raise NotImplementedError()

    def apply_docfreq(self, key, value):
        if value >= self.docfreq_threshold:
            self.docfreq[key] = value


class Repo2WeightedSet(Repo2Base):
    def __init__(self, engine, finalizer, languages=None, extractors=None, **kwargs):
        super().__init__(engine, finalizer, languages, **kwargs)
        self.extractors = extractors

    def process_uast(self, uast):
        bag = {}
        for extractor in self.extractors:
            bag.update(extractor.extract(uast.uast))
        yield uast.file_hash, bag


class Repo2DocFreq(Repo2Base):
    NDOCS_KEY = -1, 0

    def __init__(self, engine, languages=None, extractors=None, **kwargs):
        super().__init__(engine, self, languages, **kwargs)
        self.extractors = extractors

    def process_uast(self, uast):
        yield self.NDOCS_KEY, 1
        for i, extractor in self.extractors:
            for k in extractor.inspect():
                yield (i, k), 1

    def prepare_process_uast(self):
        NDOCS_KEY = copy.deepcopy(self.NDOCS_KEY)
        extractors = copy.deepcopy(self.extractors)

        def process_uast(uast):
            yield NDOCS_KEY, 1
            for i, extractor in enumerate(extractors):
                for k in extractor().inspect(uast):
                    yield (i, k), 1
        return process_uast

    def __call__(self, processed):
        reduced = processed.countByKey()
        ndocs = None
        for (i, key), value in reduced.items():
            if (i, key) == self.NDOCS_KEY:
                ndocs = value
            self.extractors[i].apply_docfreq(key, value)
        for extractor in self.extractors:
            extractor.ndocs = ndocs


@register_extractor
class IdentifiersBagExtractor(BagsExtractor):
    NAME = "id"

    class NoopTokenParser:
        def process_token(self, token):
            yield token

    def __init__(self, docfreq_threshold=None, split_stem=False):
        super().__init__(docfreq_threshold)
        self.id2bag_extract = UastIds2Bag(
            self.docfreq, self.NoopTokenParser() if not split_stem else None)
        self.id2bag_inspect = UastIds2Bag(
            None, self.NoopTokenParser() if not split_stem else None)

    def extract(self, uast):
        bag = self.id2bag_extract.uast_to_bag(uast)
        log = numpy.log
        for key, val in bag.items():
            yield key, log(1 + val) * log(self.ndocs / self.docfreq[key])

    def inspect(self, uast):
        bag = self.id2bag_inspect.uast_to_bag(uast)
        for key in bag:
            yield key


class BagsFinalizer(PickleableLogger):
    CHUNK_SIZE = 8 * 1000 * 1000 * 1000

    def __init__(self, extractors, finalizer, **kwargs):
        super().__init__(**kwargs)
        self.extractors = extractors
        self.finalizer = finalizer
        self.keys = {}

    def __call__(self, processed):
        avglen = processed.mapValues(len).mean()
        self._log.info("Average bag length: %.1f", avglen)
        keys = list(itertools.chain.from_iterable(e.docfreq for e in self.extractors))
        keys.sort()
        self.keys = {k: i for i, k in enumerate(keys)}
        del keys
        ndocs = self.extractors[0].ndocs
        chunklen = self.CHUNK_SIZE // (2 * 4 * avglen)
        nparts = ndocs // chunklen + 1
        self._log.info("chunk %d\tparts %d", chunklen, nparts)
        indexed = processed.mapValues(self.bag2row)
        self.finalizer(indexed)

    def bag2row(self, bag):
        data = numpy.zeros(len(bag), dtype=numpy.float32)
        indices = numpy.zeros(len(bag), dtype=numpy.int32)
        for i, (k, v) in enumerate(sorted((self.keys[k], v) for k, v in bag.items())):
            data[i] = v
            indices[i] = k
        return data, indices
