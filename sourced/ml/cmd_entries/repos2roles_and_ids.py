import logging
from uuid import uuid4

from sourced.ml.extractors.roles_and_ids import RolesAndIdsExtractor
from sourced.ml.transformers import Cacher, CsvSaver, Rower, create_uast_source, \
    UastDeserializer, UastRow2Document, Uast2BagFeatures
from sourced.ml.utils.engine import pause, pipeline_graph


@pause
def repos2roles_and_ids_entry(args):
    log = logging.getLogger("repos2roles_and_ids")
    session_name = "repos2roles_and_ids-%s" % uuid4()
    root, start_point = create_uast_source(args, session_name)

    start_point \
        .link(UastRow2Document()) \
        .link(Cacher.maybe(args.persist)) \
        .link(UastDeserializer()) \
        .link(Uast2BagFeatures([RolesAndIdsExtractor(args.split)])) \
        .link(Rower(lambda x: dict(identifier=x[0][0], role=x[1]))) \
        .link(CsvSaver(args.output)) \
        .execute()

    pipeline_graph(args, log, root)
