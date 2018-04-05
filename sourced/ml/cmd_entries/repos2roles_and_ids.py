from uuid import uuid4

from sourced.ml.extractors.roles_and_ids import RolesAndIdsExtractor
from sourced.ml.transformers import Ignition, UastExtractor, UastDeserializer, \
    HeadFiles, Uast2BagFeatures, Cacher, UastRow2Document, CsvSaver, LanguageSelector
from sourced.ml.transformers.basic import Rower
from sourced.ml.utils import create_engine
from sourced.ml.utils.engine import pause


@pause
def repos2roles_and_ids_entry(args):
    engine = create_engine("repos2roles_and_ids-%s" % uuid4(), **args.__dict__)

    Ignition(engine, explain=args.explain) \
        .link(HeadFiles()) \
        .link(LanguageSelector(languages=args.languages)) \
        .link(UastExtractor()) \
        .link(UastRow2Document()) \
        .link(Cacher.maybe(args.persist)) \
        .link(UastDeserializer()) \
        .link(Uast2BagFeatures([RolesAndIdsExtractor(args.split)])) \
        .link(Rower(lambda x: dict(identifier=x[0][0], role=x[1]))) \
        .link(CsvSaver(args.output)) \
        .execute()
