import logging
from uuid import uuid4

from sourced.ml.extractors import IdentifierDistance
from sourced.ml.transformers import Cacher, CsvSaver, Rower, create_uast_source, \
    UastDeserializer, UastRow2Document, Uast2BagFeatures
from sourced.ml.utils import pause, pipeline_graph


@pause
def repos2id_distance_entry(args):
    log = logging.getLogger("repos2id_distance")
    session_name = "repos2id_distance-%s" % uuid4()
    extractors = [IdentifierDistance(args.split, args.type, args.max_distance)]
    root, start_point = create_uast_source(args, session_name)

    start_point \
        .link(UastRow2Document()) \
        .link(Cacher.maybe(args.persist)) \
        .link(UastDeserializer()) \
        .link(Uast2BagFeatures(extractors)) \
        .link(Rower(lambda x: dict(identifier1=x[0][0][0],
                                   identifier2=x[0][0][1],
                                   distance=x[1]))) \
        .link(CsvSaver(args.output)) \
        .execute()
    pipeline_graph(args, log, root)
