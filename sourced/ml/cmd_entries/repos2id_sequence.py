import logging
from uuid import uuid4

from sourced.ml.extractors import IdSequenceExtractor
from sourced.ml.transformers import Cacher, CsvSaver, Rower, create_uast_source, \
    UastDeserializer, UastRow2Document, Uast2BagFeatures
from sourced.ml.utils import pause, pipeline_graph


@pause
def repos2id_sequence_entry(args):
    log = logging.getLogger("repos2id_sequence")
    session_name = "repos2id_distance-%s" % uuid4()
    extractors = [IdSequenceExtractor(args.split)]
    root, start_point = create_uast_source(args, session_name)
    if not args.skip_docname:
        mapper = Rower(lambda x: dict(document=x[0][1],
                                      identifiers=x[0][0]))
    else:
        mapper = Rower(lambda x: dict(identifiers=x[0][0]))

    start_point \
        .link(UastRow2Document()) \
        .link(Cacher.maybe(args.persist)) \
        .link(UastDeserializer()) \
        .link(Uast2BagFeatures(extractors)) \
        .link(mapper) \
        .link(CsvSaver(args.output)) \
        .execute()
    pipeline_graph(args, log, root)
