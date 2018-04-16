import logging
from uuid import uuid4

from sourced.ml.extractors import IdSequenceExtractor
from sourced.ml.transformers import Ignition, UastExtractor, UastDeserializer, \
    HeadFiles, Uast2BagFeatures, Cacher, UastRow2Document, CsvSaver, LanguageSelector
from sourced.ml.transformers.basic import Rower
from sourced.ml.utils import create_engine
from sourced.ml.utils.engine import pause


@pause
def repos2id_sequence_entry(args):
    log = logging.getLogger("repos2id_distance")
    engine = create_engine("repos2id_distance-%s" % uuid4(), **args.__dict__)
    extractors = [IdSequenceExtractor(args.split)]
    if not args.skip_docname:
        mapper = Rower(lambda x: dict(document=x[0][1],
                                      identifiers=x[0][0]))
    else:
        mapper = Rower(lambda x: dict(identifiers=x[0][0]))

    ignition = Ignition(engine, explain=args.explain)
    ignition \
        .link(HeadFiles()) \
        .link(LanguageSelector(languages=args.languages)) \
        .link(UastExtractor()) \
        .link(UastRow2Document()) \
        .link(Cacher.maybe(args.persist)) \
        .link(UastDeserializer()) \
        .link(Uast2BagFeatures(extractors)) \
        .link(mapper) \
        .link(CsvSaver(args.output)) \
        .execute()
