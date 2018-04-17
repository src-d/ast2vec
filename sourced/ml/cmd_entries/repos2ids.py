import logging
from uuid import uuid4

from sourced.ml.transformers import ContentToIdentifiers, CsvSaver, IdentifiersToDataset, \
    create_uast_source
from sourced.ml.utils.engine import pause, pipeline_graph


@pause
def repos2ids_entry(args):
    log = logging.getLogger("repos2ids")
    session_name = "repos2ids-%s" % uuid4()
    root, start_point = create_uast_source(args, session_name, extract_uast=False)

    start_point \
        .link(ContentToIdentifiers(args.split)) \
        .link(IdentifiersToDataset(args.idfreq)) \
        .link(CsvSaver(args.output)) \
        .execute()
    pipeline_graph(args, log, root)
