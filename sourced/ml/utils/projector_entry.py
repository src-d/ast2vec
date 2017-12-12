import logging

import numpy

from sourced.ml.models import Id2Vec


def projector_entry(args):
    MAX_TOKENS = 10000  # hardcoded in Tensorflow Projector

    log = logging.getLogger("id2vec_projector")
    id2vec = Id2Vec(log_level=args.log_level).load(source=args.input)
    if args.df:
        from sourced.ml.models import DocumentFrequencies
        df = DocumentFrequencies(log_level=args.log_level).load(source=args.df)
    else:
        df = None
    if len(id2vec) < MAX_TOKENS:
        tokens = numpy.arange(len(id2vec), dtype=int)
        if df is not None:
            freqs = [df.get(id2vec.tokens[i], 0) for i in tokens]
        else:
            freqs = None
    else:
        if df is not None:
            log.info("Filtering tokens through docfreq")
            items = []
            for token, idx in id2vec.items():
                try:
                    items.append((df[token], idx))
                except KeyError:
                    continue
            log.info("Sorting")
            items.sort(reverse=True)
            tokens = [i[1] for i in items[:MAX_TOKENS]]
            freqs = [i[0] for i in items[:MAX_TOKENS]]
        else:
            log.warning("You have not specified --df => picking random %d tokens", MAX_TOKENS)
            numpy.random.seed(777)
            tokens = numpy.random.choice(
                numpy.arange(len(id2vec), dtype=int), MAX_TOKENS, replace=False)
            freqs = None
    log.info("Gathering the embeddings")
    embeddings = numpy.vstack([id2vec.embeddings[i] for i in tokens])
    tokens = [id2vec.tokens[i] for i in tokens]
    labels = ["subtoken"]
    if freqs is not None:
        labels.append("docfreq")
        tokens = list(zip(tokens, (str(i) for i in freqs)))
    import sourced.ml.utils.projector as projector
    projector.present_embeddings(args.output, not args.no_browser, labels, tokens, embeddings)
    if not args.no_browser:
        projector.wait()
