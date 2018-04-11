import logging
from uuid import uuid4

from sourced.ml.extractors import create_extractors_from_args
from sourced.ml.models import OrderedDocumentFrequencies, QuantizationLevels
from sourced.ml.transformers import BagFeatures2DocFreq, BagFeatures2TermFreq, BOWWriter, \
    Cacher, HeadFiles, Indexer, Moder, create_uast_source, TFIDF, UastDeserializer, \
    UastRow2Document, Uast2Quant, Uast2BagFeatures
from sourced.ml.utils import pause, pipeline_graph


@pause
def repos2bow_entry_template(args, select=HeadFiles, cache_hook=None, save_hook=None):
    log = logging.getLogger("repos2bow")
    extractors = create_extractors_from_args(args)
    session_name = "repos2bow-%s" % uuid4()
    root, start_point = create_uast_source(args, session_name, select=select)

    uast_extractor = start_point.link(Moder(args.mode)).link(Cacher.maybe(args.persist))

    if cache_hook is not None:
        uast_extractor.link(cache_hook()).execute()
    # We link UastRow2Document after Cacher here because cache_hook() may want to have all possible
    # Row items.
    uast_extractor = uast_extractor.link(UastRow2Document())
    log.info("Extracting UASTs and indexing documents...")
    document_indexer = Indexer(Uast2BagFeatures.Columns.document)
    uast_extractor.link(document_indexer).execute()
    ndocs = len(document_indexer)
    log.info("Number of documents: %d", ndocs)
    uast_extractor = uast_extractor.link(UastDeserializer())
    quant = Uast2Quant(extractors)
    uast_extractor.link(quant).execute()
    if quant.levels:
        log.info("Writing quantization levels to %s", args.quant)
        QuantizationLevels().construct(quant.levels).save(args.quant)
    uast_extractor = uast_extractor \
        .link(Uast2BagFeatures(extractors)) \
        .link(Cacher.maybe(args.persist))
    log.info("Calculating the document frequencies...")
    df = uast_extractor.link(BagFeatures2DocFreq()).execute()
    log.info("Writing docfreq to %s", args.docfreq)
    df_model = OrderedDocumentFrequencies() \
        .construct(ndocs, df) \
        .prune(args.min_docfreq) \
        .greatest(args.vocabulary_size) \
        .save(args.docfreq)
    bags_writer = uast_extractor \
        .link(BagFeatures2TermFreq()) \
        .link(TFIDF(df_model)) \
        .link(document_indexer) \
        .link(Indexer(Uast2BagFeatures.Columns.token, df_model.order))
    if save_hook is not None:
        bags_writer = bags_writer.link(save_hook())
    bags_writer.link(BOWWriter(document_indexer, df_model, args.bow, args.batch)) \
        .execute()
    pipeline_graph(args, log, root)


def repos2bow_entry(args):
    return repos2bow_entry_template(args)
