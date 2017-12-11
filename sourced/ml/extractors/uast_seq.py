from sourced.ml.algorithms import UastSeq2Bag
from sourced.ml.extractors import BagsExtractor, register_extractor, get_names_from_kwargs


@register_extractor
class UastSeqBagExtractor(BagsExtractor):
    NAME = "uast2seq"
    NAMESPACE = "s."
    OPTS = dict(get_names_from_kwargs(UastSeq2Bag.__init__))

    def __init__(self, docfreq_threshold=None, **kwargs):
        super().__init__(docfreq_threshold)
        self._log.debug("__init__ %s", kwargs)
        self.uast2bag = UastSeq2Bag(**kwargs)

    def uast_to_bag(self, uast):
        return self.uast2bag.uast_to_bag(uast)
