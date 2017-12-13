import codecs
import os

from sourced.ml.algorithms.uast_ids_to_bag import UastIds2Bag
from sourced.ml.extractors import BagsExtractor, register_extractor


@register_extractor
class LiteralsBagExtractor(BagsExtractor):
    NAME = "lit"
    NAMESPACE = "l."
    OPTS = BagsExtractor.OPTS.copy()

    class HashedTokenParser:
        def process_token(self, token):
            yield codecs.encode((hash(token) & 0xffffffffffffffff).to_bytes(8, "little"),
                                "hex_codec").decode()

    def __init__(self, docfreq_threshold=None):
        super().__init__(docfreq_threshold)
        self.id2bag = UastIds2Bag(None, self.HashedTokenParser())

    def uast_to_bag(self, uast):
        if os.getenv("PYTHONHASHSEED", "random") == "random":
            raise RuntimeError("PYTHONHASHSEED must be set")
        return self.id2bag.uast_to_bag(uast, roles_filter="//*[@roleLiteral]")
