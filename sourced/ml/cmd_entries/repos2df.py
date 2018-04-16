import logging
from uuid import uuid4

from sourced.ml.extractors import create_extractors_from_args
from sourced.ml.models import OrderedDocumentFrequencies, QuantizationLevels
from sourced.ml.transformers import Ignition, UastExtractor, UastDeserializer, LanguageSelector, \
    BagFeatures2DocFreq, HeadFiles, Uast2BagFeatures, Counter, Cacher, Uast2Quant, UastRow2Document
from sourced.ml.utils import create_engine
from sourced.ml.utils.engine import pipeline_graph, pause


@pause
def repos2df_entry(args):
    log = logging.getLogger("repos2df")
    engine = create_engine("repos2df-%s" % uuid4(), **args.__dict__)
    extractors = create_extractors_from_args(args)

    ignition = Ignition(engine, explain=args.explain)
    uast_extractor = ignition \
        .link(HeadFiles()) \
        .link(LanguageSelector(languages=args.languages)) \
        .link(UastExtractor()) \
        .link(UastRow2Document()) \
        .link(Cacher.maybe(args.persist))
    log.info("Extracting UASTs...")
    ndocs = uast_extractor.link(Counter()).execute()
    log.info("Number of documents: %d", ndocs)
    uast_extractor = uast_extractor.link(UastDeserializer())
    quant = Uast2Quant(extractors)
    uast_extractor.link(quant).execute()
    if quant.levels:
        log.info("Writing quantization levels to %s", args.quant)
        QuantizationLevels().construct(quant.levels).save(args.quant)
    df = uast_extractor \
        .link(Uast2BagFeatures(extractors)) \
        .link(BagFeatures2DocFreq()) \
        .execute()
    log.info("Writing %s", args.docfreq)
    OrderedDocumentFrequencies().construct(ndocs, df).save(args.docfreq)
    pipeline_graph(args, log, ignition)
