from sourced.ml.algorithms import UastNode2Bag, Quantization
from sourced.ml.extractors import BagsExtractor, register_extractor


@register_extractor
class RoleTypeChildrenBagExtractor(BagsExtractor):
    NAME = "rolechildren"
    NAMESPACE = "rc."
    OPTS = BagsExtractor.OPTS.copy()

    def __init__(self, docfreq_threshold=None):
        super().__init__(docfreq_threshold)
        self.role_children2bag = UastNode2Bag(children2ind=Quantization())

    def uast_to_bag(self, uast):
        return self.role_children2bag.uast_to_bag(uast, feature="roles")
