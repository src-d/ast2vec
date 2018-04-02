import argparse
import logging
import os
from uuid import uuid4

from sourced.ml.extractors import create_extractors_from_args
from sourced.ml.models import OrderedDocumentFrequencies, QuantizationLevels
from sourced.ml.transformers import Ignition, UastExtractor, UastDeserializer, Uast2Quant, \
    BagFeatures2DocFreq, BagFeatures2TermFreq, Uast2BagFeatures, HeadFiles, TFIDF, Cacher, \
    Indexer, UastRow2Document, BOWWriter, Moder, create_parquet_loader
from sourced.ml.utils import create_engine
from sourced.ml.utils.engine import pipeline_graph, pause


def add_bow_args(my_parser: argparse.ArgumentParser):
    my_parser.add_argument(
        "--bow", required=True, help="[OUT] The path to the Bag-Of-Words model.")
    my_parser.add_argument(
        "--batch", default=BOWWriter.DEFAULT_CHUNK_SIZE, type=int,
        help="The maximum size of a single BOW file in bytes.")
    my_parser.add_argument(
        "--parquet", action="store_true", help="If it's parquet input.")


@pause
def repos2bow_entry_template(args, select=HeadFiles, cache_hook=None):
    log = logging.getLogger("repos2bow")
    extractors = create_extractors_from_args(args)
    session_name = "repos2bow-%s" % uuid4()
    if args.parquet:
        start_point = create_parquet_loader(session_name, **args.__dict__)
        root = start_point
    else:
        engine = create_engine(session_name, **args.__dict__)
        root = engine

        start_point = Ignition(engine, explain=args.explain) \
            .link(select()) \
            .link(UastExtractor(languages=args.languages))

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
    dirs = os.path.split(args.docfreq)[0]
    if dirs:
        os.makedirs(dirs, exist_ok=True)
    dirs = os.path.split(args.bow)[0]
    if dirs:
        os.makedirs(dirs, exist_ok=True)
    df_model = OrderedDocumentFrequencies() \
        .construct(ndocs, df) \
        .prune(args.min_docfreq) \
        .greatest(args.vocabulary_size) \
        .save(args.docfreq)
    uast_extractor \
        .link(BagFeatures2TermFreq()) \
        .link(TFIDF(df_model)) \
        .link(document_indexer) \
        .link(Indexer(Uast2BagFeatures.Columns.token, df_model.order)) \
        .link(BOWWriter(document_indexer, df_model, args.bow, args.batch)) \
        .execute()
    pipeline_graph(args, log, root)


def repos2bow_entry(args):
    return repos2bow_entry_template(args)
