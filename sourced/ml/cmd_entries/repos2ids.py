from uuid import uuid4

from sourced.ml.transformers import Ignition, ContentToIdentifiers, \
    LanguageSelector, IdentifiersToDataset, HeadFiles, Cacher, CsvSaver
from sourced.ml.utils import create_engine
from sourced.ml.utils.engine import pause


@pause
def repos2ids_entry(args):
    engine = create_engine("repos2ids-%s" % uuid4(), **args.__dict__)

    Ignition(engine) \
        .link(HeadFiles()) \
        .link(LanguageSelector(languages=args.languages)) \
        .link(ContentToIdentifiers(args.split)) \
        .link(Cacher.maybe(args.persist)) \
        .link(IdentifiersToDataset(args.idfreq)) \
        .link(CsvSaver(args.output)) \
        .execute()
