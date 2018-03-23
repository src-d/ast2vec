import logging
from uuid import uuid4

from pyspark import Row

from sourced.ml.extractors.roles_and_ids import RolesAndIdsExtractor
from sourced.ml.transformers import Ignition, UastExtractor, UastDeserializer, \
    HeadFiles, Uast2BagFeatures, Cacher, UastRow2Document, Mapper, CsvSaver
from sourced.ml.utils import create_engine
from sourced.ml.utils.engine import pause


@pause
def repos2roles_and_ids_entry(args):
    log = logging.getLogger("repos2df")
    engine = create_engine("repos2df-%s" % uuid4(), **args.__dict__)
    extractors = [RolesAndIdsExtractor(args.split)]

    ignition = Ignition(engine, explain=args.explain)
    ignition \
        .link(HeadFiles()) \
        .link(UastExtractor(languages=args.languages)) \
        .link(UastRow2Document()) \
        .link(Cacher.maybe(args.persist)) \
        .link(UastDeserializer()) \
        .link(Uast2BagFeatures(extractors)) \
        .link(Mapper(lambda x: Row(**{"identifier": x[0][0], "role": x[1]}))) \
        .link(CsvSaver(args.output)) \
        .execute()
