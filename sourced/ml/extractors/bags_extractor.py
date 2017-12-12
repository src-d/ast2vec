import numpy

from sourced.ml.utils import PickleableLogger


class BagsExtractor(PickleableLogger):
    """
    Converts a single UAST into the weighted set (dictionary), where elements are strings
    and the values are floats. The derived classes must implement uast_to_bag().
    """
    DEFAULT_DOCFREQ_THRESHOLD = 5
    NAMESPACE = None  # the beginning of each element in the bag
    OPTS = {}  # cmdline args which are passed into __init__()

    def __init__(self, docfreq_threshold=None, **kwargs):
        """
        :param docfreq_threshold: The minimum number of occurrences of an element to be included \
                                  into the bag
        """
        super().__init__(**kwargs)
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

    def _get_log_name(self):
        return type(self).__name__

    def extract(self, uast):
        ndocs = self.ndocs
        docfreq = self.docfreq
        log = numpy.log
        for key, val in self.uast_to_bag(uast).items():
            key = self.NAMESPACE + key
            try:
                yield key, log(1 + val) * log(ndocs / docfreq[key])
            except KeyError:
                # docfreq_threshold
                continue

    def inspect(self, uast):
        try:
            bag = self.uast_to_bag(uast)
        except RuntimeError as e:
            raise ValueError(str(uast)) from e
        for key in bag:
            yield self.NAMESPACE + key

    def apply_docfreq(self, key, value):
        if value >= self.docfreq_threshold:
            if not isinstance(key, str):
                raise TypeError("key is %s" % type(key))
            self.docfreq[str(key)] = value

    @classmethod
    def get_kwargs_fromcmdline(cls, args):
        prefix = cls.NAME + "_"
        result = {}
        for k, v in args.__dict__.items():
            if k.startswith(prefix):
                result[k[len(prefix):]] = v
        return result

    def uast_to_bag(self, uast):
        raise NotImplementedError()
