import logging
from uuid import uuid4

from sourced.ml.transformers import ContentToIdentifiers, CsvSaver, IdentifiersToDataset, \
    Ignition, HeadFiles, LanguageSelector
from sourced.ml.utils.engine import create_engine, pause, pipeline_graph


@pause
def repos2ids_entry(args):
    log = logging.getLogger("repos2ids")
    session_name = "repos2ids-%s" % uuid4()
    root = create_engine(session_name, **args.__dict__)
    Ignition(root, explain=args.explain) \
        .link(HeadFiles()) \
        .link(LanguageSelector(languages=["null"], blacklist=True)) \
        .link(ContentToIdentifiers(args.split)) \
        .link(IdentifiersToDataset(args.idfreq)) \
        .link(CsvSaver(args.output)) \
        .execute()
    pipeline_graph(args, log, root)
