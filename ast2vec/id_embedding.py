import logging
import os
import shutil
import sys
from argparse import Namespace
from collections import defaultdict

import numpy
from modelforge.progress_bar import progress_bar
from scipy.sparse import csr_matrix
import tensorflow as tf

import ast2vec.swivel as swivel
from ast2vec.coocc import Cooccurrences
from ast2vec.df import DocumentFrequencies
from ast2vec.id2vec import Id2Vec
from ast2vec.token_parser import TokenParser
from ast2vec.repo2.base import Transformer


class PreprocessTransformer(Transformer):
    vocabulary_size = 1 << 17
    shard_size = 4096

    def __init__(self, vocabulary_size=None, shard_size=None):
        super(PreprocessTransformer, self).__init__()
        if vocabulary_size is not None:
            self.vocabulary_size = vocabulary_size
        if shard_size is not None:
            self.shard_size = shard_size

    def transform(self, X, output, df=None, vocabulary_size=None,
                  shard_size=None):
        if vocabulary_size is not None:
            self.vocabulary_size = vocabulary_size
        if shard_size is not None:
            self.shard_size = shard_size

        if isinstance(X, str):
            X = [X]

        args = Namespace(vocabulary_size=self.vocabulary_size,
                         input=X, df=df, shard_size=self.shard_size,
                         output=output)
        preprocess(args)

    def _get_log_name(self):
        return "id_preprocess"


def preprocess(args):
    """
    Loads co-occurrence matrices for several repositories and generates the
    document frequencies and the Swivel protobuf dataset.

    :param args: :class:`argparse.Namespace` with "input", "vocabulary_size", \
                 "shard_size", "df" and "output".
    :return: None
    """
    log = logging.getLogger("preproc")
    log.info("Scanning the inputs...")
    inputs = []
    for i in args.input:
        if os.path.isdir(i):
            inputs.extend([os.path.join(i, f) for f in os.listdir(i)])
        else:
            inputs.append(i)
    log.info("Reading word indices from %d files...", len(inputs))
    all_words = defaultdict(int)
    skipped = 0
    for i, path in progress_bar(enumerate(inputs), log, expected_size=len(inputs)):
        try:
            model = Cooccurrences().load(source=path)
        except ValueError:
            skipped += 1
            log.warning("Skipped %s", path)
            continue
        for w in model.tokens:
            all_words[w] += 1
    vs = args.vocabulary_size
    if len(all_words) < vs:
        vs = len(all_words)
    sz = args.shard_size
    if vs < sz:
        raise ValueError(
            "vocabulary_size={0} is less than shard_size={1}. "
            "You should specify smaller shard_size "
            "(pass shard_size={0} argument).".format(vs, sz))
    vs -= vs % sz
    log.info("Effective vocabulary size: %d", vs)
    log.info("Truncating the vocabulary...")
    words = numpy.array(list(all_words.keys()))
    freqs = numpy.array(list(all_words.values()), dtype=numpy.int64)
    del all_words
    chosen_indices = numpy.argpartition(
        freqs, len(freqs) - vs)[len(freqs) - vs:]
    chosen_freqs = freqs[chosen_indices]
    chosen_words = words[chosen_indices]
    border_freq = chosen_freqs.min()
    border_mask = chosen_freqs == border_freq
    border_num = border_mask.sum()
    border_words = words[freqs == border_freq]
    border_words = numpy.sort(border_words)
    chosen_words[border_mask] = border_words[:border_num]
    del words
    del freqs
    log.info("Sorting the vocabulary...")
    sorted_indices = numpy.argsort(chosen_words)
    chosen_freqs = chosen_freqs[sorted_indices]
    chosen_words = chosen_words[sorted_indices]
    word_indices = {w: i for i, w in enumerate(chosen_words)}
    if args.df is not None:
        log.info("Writing the document frequencies to %s...", args.df)
        model = DocumentFrequencies()
        model.construct(docs=len(inputs) - skipped, tokens=chosen_words, freqs=chosen_freqs)
        model.save(args.df)
    del chosen_freqs

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    with open(os.path.join(args.output, "row_vocab.txt"), "w") as out:
        out.write('\n'.join(chosen_words))
    log.info("Saved row_vocab.txt...")
    shutil.copyfile(os.path.join(args.output, "row_vocab.txt"),
                    os.path.join(args.output, "col_vocab.txt"))
    log.info("Saved col_vocab.txt...")

    del chosen_words
    log.info("Combining individual co-occurrence matrices...")
    ccmatrix = csr_matrix((vs, vs), dtype=numpy.int64)
    for i, path in progress_bar(enumerate(inputs), log, expected_size=len(inputs)):
        try:
            model = Cooccurrences().load(path)
        except ValueError:
            log.warning("Skipped %s", path)
            continue
        if len(model) == 0:
            log.warning("Skipped %s", path)
            continue
        matrix = _extract_coocc_matrix(ccmatrix.shape, word_indices, model)
        # Stage 5 - simply add this converted matrix to the global one
        ccmatrix += matrix

    log.info("Planning the sharding...")
    bool_sums = ccmatrix.indptr[1:] - ccmatrix.indptr[:-1]
    with open(os.path.join(args.output, "row_sums.txt"), "w") as out:
        out.write('\n'.join(map(str, bool_sums.tolist())))
    log.info("Saved row_sums.txt...")
    shutil.copyfile(os.path.join(args.output, "row_sums.txt"),
                    os.path.join(args.output, "col_sums.txt"))
    log.info("Saved col_sums.txt...")
    reorder = numpy.argsort(-bool_sums)
    log.info("Writing the shards...")
    os.makedirs(args.output, exist_ok=True)
    nshards = vs // args.shard_size
    for row in progress_bar(range(nshards), log, expected_size=nshards):
        for col in range(nshards):
            def _int64s(xs):
                return tf.train.Feature(
                    int64_list=tf.train.Int64List(value=list(xs)))

            def _floats(xs):
                return tf.train.Feature(
                    float_list=tf.train.FloatList(value=list(xs)))

            indices_row = reorder[row::nshards]
            indices_col = reorder[col::nshards]
            shard = ccmatrix[indices_row][:, indices_col].tocoo()

            example = tf.train.Example(features=tf.train.Features(feature={
                "global_row": _int64s(indices_row),
                "global_col": _int64s(indices_col),
                "sparse_local_row": _int64s(shard.row),
                "sparse_local_col": _int64s(shard.col),
                "sparse_value": _floats(shard.data)}))

            with open(os.path.join(args.output,
                                   "shard-%03d-%03d.pb" % (row, col)),
                      "wb") as out:
                out.write(example.SerializeToString())
    log.info("Success")


def _extract_coocc_matrix(global_shape, word_indices, model):
    # Stage 1 - extract the tokens, map them to the global vocabulary
    indices = []
    mapped_indices = []
    for i, w in enumerate(model.tokens):
        gi = word_indices.get(w)
        if gi is not None:
            indices.append(i)
            mapped_indices.append(gi)
    indices = numpy.array(indices)
    mapped_indices = numpy.array(mapped_indices)
    # Stage 2 - sort the matched tokens by the index in the vocabulary
    order = numpy.argsort(mapped_indices)
    indices = indices[order]
    mapped_indices = mapped_indices[order]
    # Stage 3 - produce the csr_matrix with the matched tokens **only**
    matrix = model.matrix.tocsr()[indices][:, indices]
    # Stage 4 - convert this matrix to the global (ccmatrix) coordinates
    csr_indices = matrix.indices
    for i, v in enumerate(csr_indices):
        # Here we use the fact that indices and mapped_indices are in the same order
        csr_indices[i] = mapped_indices[v]
    csr_indptr = matrix.indptr
    new_indptr = [0]
    for i, v in enumerate(mapped_indices):
        prev_ptr = csr_indptr[i]
        ptr = csr_indptr[i + 1]

        # Handle missing rows
        prev = (mapped_indices[i - 1] + 1) if i > 0 else 0
        for z in range(prev, v):
            new_indptr.append(prev_ptr)

        new_indptr.append(ptr)
    for z in range(mapped_indices[-1] + 1, global_shape[0]):
        new_indptr.append(csr_indptr[-1])
    matrix.indptr = numpy.array(new_indptr)
    matrix._shape = global_shape
    return matrix


class SwivelTransformer(Transformer):
    def transform(self, **kwargs):
        flags = type(swivel.FLAGS)()
        flags.__dict__ = swivel.FLAGS.__dict__.copy()

        for key, val in kwargs.items():
            if val is not None:
                setattr(flags, key, val)

        run_swivel(flags)

    def _get_log_name(self):
        return "id_swivel"


def run_swivel(args):
    """
    Trains the Swivel model. Wraps swivel.py, adapted from
    https://github.com/vmarkovtsev/models/blob/master/swivel/swivel.py

    :param args: :class:`argparse.Namespace` identical to \
                 :class:`tf.app.flags`.
    :return: None
    """
    swivel.FLAGS = args
    logging.getLogger("tensorflow").handlers.clear()
    swivel.main(args)


class PostprocessTransformer(Transformer):
    def transform(self, swivel_output_directory, result):
        """
        Merges row and column embeddings produced by Swivel and writes the
        Id2Vec model.

        :param swivel_output_directory: directory that contains files after swivel training. The \
                                        files are read from this directory and the model is \
                                        written to the 'result'.
        :param result: file to store results
        :return: None
        """
        args = Namespace(swivel_output_directory=swivel_output_directory,
                         result=result)
        postprocess(args)

    def _get_log_name(self):
        return "id_postprocess"


def postprocess(args):
    """
    Merges row and column embeddings produced by Swivel and writes the Id2Vec
    model.

    :param args: :class:`argparse.Namespace` with "swivel_output_directory" \
                 and "result". The text files are read from \
                 `swivel_output_directory` and the model is written to \
                 `result`.
    :return: None
    """
    log = logging.getLogger("postproc")
    log.info("Parsing the embeddings at %s...", args.swivel_output_directory)
    tokens = []
    embeddings = []
    swd = args.swivel_output_directory
    with open(os.path.join(swd, "row_embedding.tsv")) as frow:
        with open(os.path.join(swd, "col_embedding.tsv")) as fcol:
            for i, (lrow, lcol) in enumerate(zip(frow, fcol)):
                if i % 10000 == (10000 - 1):
                    sys.stdout.write("%d\r" % (i + 1))
                    sys.stdout.flush()
                prow, pcol = (l.split("\t", 1) for l in (lrow, lcol))
                assert prow[0] == pcol[0]
                tokens.append(prow[0][:TokenParser.MAX_TOKEN_LENGTH])
                erow, ecol = \
                    (numpy.fromstring(p[1], dtype=numpy.float32, sep="\t")
                     for p in (prow, pcol))
                embeddings.append((erow + ecol) / 2)
    log.info("Generating numpy arrays...")
    embeddings = numpy.array(embeddings, dtype=numpy.float32)
    log.info("Writing %s...", args.result)
    model = Id2Vec()
    model.construct(embeddings=embeddings, tokens=tokens)
    model.save(args.result)
