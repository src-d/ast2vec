import logging
from uuid import uuid4

from sourced.ml.extractors import create_extractors_from_args
from sourced.ml.models import DocumentFrequencies
from sourced.ml.transformers import (
    BagFeatures2TermFreq, BOWWriter, Cacher, Collector, create_uast_source, Distinct,
    FieldsSelector, Indexer, Moder, PartitionSelector, Repartitioner, TFIDF, Transformer,
    Uast2BagFeatures, UastDeserializer, UastRow2Document)
from sourced.ml.utils.docfreq import create_or_load_ordered_df
from sourced.ml.utils.engine import EngineConstants, pause, pipeline_graph
from sourced.ml.utils.quant import create_or_apply_quant


@pause
def repos2bow_template(args, cache_hook: Transformer = None,
                       save_hook: Transformer = None):

    log = logging.getLogger("repos2bow")
    extractors = create_extractors_from_args(args)
    session_name = "repos2bow-%s" % uuid4()
    root, start_point = create_uast_source(args, session_name)
    log.info("Loading the document index from %s ...", args.cached_index_path)
    docfreq = DocumentFrequencies().load(source=args.cached_index_path)
    document_index = {key: int(val) for (key, val) in docfreq}

    try:
        if args.quant is not None:
            create_or_apply_quant(args.quant, extractors, None)
        df_model = create_or_load_ordered_df(args, None, None)
    except ValueError:
        return 1
    ec = EngineConstants.Columns

    if args.mode == Moder.Options.repo:
        def keymap(r):
            return r[ec.RepositoryId]
    else:
        def keymap(r):
            return r[ec.RepositoryId] + UastRow2Document.REPO_PATH_SEP + \
                r[ec.Path] + UastRow2Document.PATH_BLOB_SEP + r[ec.BlobId]

    log.info("Caching UASTs to disk after partitioning by document ...")
    start_point = start_point.link(Moder(args.mode)) \
        .link(Repartitioner.maybe(args.num_iterations, keymap=keymap)) \
        .link(Cacher.maybe("DISK_ONLY"))
    for num_part in range(args.num_iterations):
        log.info("Running job %s of %s", num_part + 1, args.num_iterations)
        selected_part = start_point \
            .link(PartitionSelector(num_part))  \
            .link(Repartitioner.maybe(args.partitions, args.shuffle)) \
            .link(Cacher.maybe(args.persist))
        if cache_hook is not None:
            selected_part.link(cache_hook()).execute()
        uast_extractor = selected_part \
            .link(UastRow2Document()) \
            .link(Cacher.maybe(args.persist))
        log.info("Collecting distinct documents ...")
        documents = uast_extractor \
            .link(FieldsSelector([Uast2BagFeatures.Columns.document])) \
            .link(Distinct()) \
            .link(Collector()) \
            .execute()
        selected_part.unpersist()
        documents = {row.document for row in documents}
        reduced_doc_index = {
            key: document_index[key] for key in document_index if key in documents}
        document_indexer = Indexer(Uast2BagFeatures.Columns.document, reduced_doc_index)
        log.info("Processing %s distinct documents", len(documents))
        bags = uast_extractor \
            .link(UastDeserializer()) \
            .link(Uast2BagFeatures(*extractors)) \
            .link(BagFeatures2TermFreq()) \
            .link(Cacher.maybe(args.persist))
        log.info("Extracting UASTs and collecting distinct tokens ...")
        tokens = bags \
            .link(FieldsSelector([Uast2BagFeatures.Columns.token])) \
            .link(Distinct()) \
            .link(Collector()) \
            .execute()
        uast_extractor.unpersist()
        tokens = {row.token for row in tokens}
        reduced_token_freq = {key: df_model[key] for key in df_model._df if key in tokens}
        reduced_token_index = {key: df_model.order[key] for key in df_model._df if key in tokens}
        log.info("Processing %s distinct tokens", len(reduced_token_freq))
        log.info("Indexing by document and token ...")
        bags_writer = bags \
            .link(TFIDF(reduced_token_freq, df_model.docs, root.engine.session.sparkContext)) \
            .link(document_indexer) \
            .link(Indexer(Uast2BagFeatures.Columns.token, reduced_token_index))
        if save_hook is not None:
            bags_writer = bags_writer \
                .link(Repartitioner.maybe(args.partitions, args.shuffle)) \
                .link(save_hook())
        bow = args.bow.split(".asdf")[0] + "_" + str(num_part + 1) + ".asdf"
        bags_writer \
            .link(Repartitioner.maybe(
                args.partitions, keymap=lambda x: x[Uast2BagFeatures.Columns.document])) \
            .link(BOWWriter(document_indexer, df_model, bow, "repos2bow", args.batch)) \
            .execute()
        bags.unpersist()
    pipeline_graph(args, log, root)


def repos2bow(args):
    return repos2bow_template(args)


def repos2bow_index_template(args):
    log = logging.getLogger("repos2bow_index")
    extractors = create_extractors_from_args(args)
    session_name = "repos2bow_index_features-%s" % uuid4()
    root, start_point = create_uast_source(args, session_name)
    uast_extractor = start_point.link(Moder(args.mode)) \
        .link(Repartitioner.maybe(args.partitions, args.shuffle)) \
        .link(UastRow2Document()) \
        .link(Cacher.maybe(args.persist))
    log.info("Extracting UASTs and indexing documents ...")
    document_indexer = Indexer(Uast2BagFeatures.Columns.document)
    uast_extractor.link(document_indexer).execute()
    document_indexer.save_index(args.cached_index_path)
    ndocs = len(document_indexer)
    log.info("Number of documents: %d", ndocs)
    uast_extractor = uast_extractor.link(UastDeserializer())
    if args.quant:
        create_or_apply_quant(args.quant, extractors, uast_extractor)
    if args.docfreq_out:
        create_or_load_ordered_df(args, ndocs, uast_extractor.link(Uast2BagFeatures(*extractors)))
    pipeline_graph(args, log, root)


def repos2bow_index(args):
    return repos2bow_index_template(args)
