import logging
from uuid import uuid4

from pyspark import Row

from sourced.ml.extractors import IdentifierDistance
from sourced.ml.transformers import Ignition, UastExtractor, UastDeserializer, \
    HeadFiles, Uast2BagFeatures, Cacher, UastRow2Document, Mapper, CsvSaver
from sourced.ml.utils import create_engine
from sourced.ml.utils.engine import pause


@pause
def repos2id_distance_entry(args):
    log = logging.getLogger("repos2id_distance")
    engine = create_engine("repos2id_distance-%s" % uuid4(), **args.__dict__)
    extractors = [IdentifierDistance(args.split, args.type, args.max_distance)]

    ignition = Ignition(engine, explain=args.explain)
    ignition \
        .link(HeadFiles()) \
        .link(UastExtractor(languages=args.languages)) \
        .link(UastRow2Document()) \
        .link(Cacher.maybe(args.persist)) \
        .link(UastDeserializer()) \
        .link(Uast2BagFeatures(extractors)) \
        .link(Mapper(lambda x: Row(**{"identifier1": x[0][0][0],
                                      "identifier2": x[0][0][1],
                                      "distance": x[1]}))) \
        .link(CsvSaver(args.output)) \
        .execute()
