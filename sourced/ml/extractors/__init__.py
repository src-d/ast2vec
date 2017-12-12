import inspect

from sourced.ml.extractors.bags_extractor import BagsExtractor
from sourced.ml.extractors.identifiers import IdentifiersBagExtractor
from sourced.ml.extractors.literals import LiteralsBagExtractor
from sourced.ml.extractors.uast_random_walk import UastRandomWalkBagExtractor
from sourced.ml.extractors.uast_seq import UastSeqBagExtractor


__extractors__ = {}


def register_extractor(cls):
    if not issubclass(cls, BagsExtractor):
        raise TypeError("%s is not an instance of %s" % (cls.__name__, BagsExtractor.__name__))
    __extractors__[cls.NAME] = cls
    return cls


def get_names_from_kwargs(f):
    for k, v in inspect.signature(f).parameters.items():
        if v.default != inspect.Parameter.empty and isinstance(
                v.default, (str, int, float, tuple)):
            yield k.replace("_", "-"), v.default