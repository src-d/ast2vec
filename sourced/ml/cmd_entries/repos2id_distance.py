import logging
from uuid import uuid4

from sourced.ml.extractors import IdentifierDistance
from sourced.ml.transformers import Ignition, UastExtractor, UastDeserializer, \
    HeadFiles, Uast2BagFeatures, Cacher, UastRow2Document, CsvSaver, LanguageSelector
from sourced.ml.transformers.basic import Rower
from sourced.ml.utils import create_engine
from sourced.ml.utils.engine import pause


@pause
def repos2id_distance_entry(args):
    engine = create_engine("repos2id_distance-%s" % uuid4(), **args.__dict__)
    extractors = [IdentifierDistance(args.split, args.type, args.max_distance)]

    Ignition(engine, explain=args.explain) \
        .link(HeadFiles()) \
        .link(LanguageSelector(languages=args.languages)) \
        .link(UastExtractor()) \
        .link(UastRow2Document()) \
        .link(Cacher.maybe(args.persist)) \
        .link(UastDeserializer()) \
        .link(Uast2BagFeatures(extractors)) \
        .link(Rower(lambda x: dict(identifier1=x[0][0][0],
                                   identifier2=x[0][0][1],
                                   distance=x[1]))) \
        .link(CsvSaver(args.output)) \
        .execute()
