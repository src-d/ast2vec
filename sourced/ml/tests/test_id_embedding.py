import argparse
import os
import subprocess
import tempfile
import unittest

import asdf
import numpy
import tensorflow as tf
from scipy.sparse import coo_matrix

from modelforge.model import split_strings, assemble_sparse_matrix
from sourced.ml.algorithms import swivel
from sourced.ml.algorithms.id_embedding import preprocess, run_swivel, postprocess, SwivelTransformer, \
    PostprocessTransformer, PreprocessTransformer
from sourced.ml.models import DocumentFrequencies, Id2Vec
from sourced.ml.tests.test_dump import captured_output


def prepare_file(path):
    """
    Check if file doesn't exist -> try to extract: path + ".gz"
    :param path: path to file
    :return: None
    """
    if not os.path.exists(path):
        subprocess.check_call(["gzip", "-dk", path + ".gz"])


def prepare_shard(dirname):
    prepare_file(os.path.join(dirname, "shard-000-000.pb"))


def prepare_postproc_files(dirname):
    for name in ("col_embedding.tsv", "row_embedding.tsv"):
        prepare_file(os.path.join(dirname, name))


def check_postproc_results(obj, id2vec_loc):
    id2vec = Id2Vec().load(source=id2vec_loc)
    obj.assertEqual(len(id2vec.tokens), obj.VOCAB)
    obj.assertEqual(id2vec.embeddings.shape, (obj.VOCAB, 50))


def check_swivel_results(obj, dirname):
    files = sorted(os.listdir(dirname))
    obj.assertEqual(files, ["col_embedding.tsv", "row_embedding.tsv"])
    with open(os.path.join(dirname, "col_embedding.tsv")) as fin:
        col_embedding = fin.read().split("\n")
    obj.assertEqual(len(col_embedding), obj.VOCAB + 1)
    with open(os.path.join(dirname, "row_embedding.tsv")) as fin:
        row_embedding = fin.read().split("\n")
    obj.assertEqual(len(row_embedding), obj.VOCAB + 1)


def default_swivel_args(tmpdir):
    args = swivel.FLAGS
    args.input_base_path = os.path.join(os.path.dirname(__file__), "swivel")
    prepare_shard(args.input_base_path)
    args.output_base_path = tmpdir
    args.embedding_size = 50
    args.num_epochs = 20
    return args


def default_preprocess_params(tmpdir, vocab):
    args = argparse.Namespace(
        output=tmpdir, df=os.path.join(tmpdir, "docfreq.asdf"),
        input=[os.path.join(os.path.dirname(__file__), "coocc")],
        vocabulary_size=vocab, shard_size=vocab)
    return args


class IdEmbeddingTests(unittest.TestCase):
    VOCAB = 4096

    def test_preprocess_bad_params(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = default_preprocess_params(tmpdir, self.VOCAB)
            args.shard_size = self.VOCAB + 1
            self.assertRaises(ValueError, lambda: preprocess(args))

    def test_preproc_transformer_logs(self):
        self.assertTrue(PreprocessTransformer()._get_log_name())

    def test_preprocess(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = default_preprocess_params(tmpdir, self.VOCAB)
            with captured_output() as (out, err, log):
                preprocess(args)
            self.assertFalse(out.getvalue())
            self.assertFalse(err.getvalue())
            self.assertIn("Skipped", log.getvalue())
            self.assertIn("error.asdf", log.getvalue())
            self.assertIn("empty_coocc.asdf", log.getvalue())
            self.assertEqual(
                sorted(os.listdir(tmpdir)),
                ["col_sums.txt", "col_vocab.txt", "docfreq.asdf", "row_sums.txt", "row_vocab.txt",
                 "shard-000-000.pb"])
            df = DocumentFrequencies().load(source=os.path.join(tmpdir, "docfreq.asdf"))
            self.assertEqual(len(df), self.VOCAB)
            self.assertEqual(df.docs, len(os.listdir(args.input[0])) - 1)
            with open(os.path.join(tmpdir, "col_sums.txt")) as fin:
                col_sums = fin.read()
            with open(os.path.join(tmpdir, "row_sums.txt")) as fin:
                row_sums = fin.read()
            self.assertEqual(col_sums, row_sums)
            with open(os.path.join(tmpdir, "col_vocab.txt")) as fin:
                col_vocab = fin.read()
            with open(os.path.join(tmpdir, "row_vocab.txt")) as fin:
                row_vocab = fin.read()
            self.assertEqual(col_vocab, row_vocab)
            self.assertEqual(row_vocab.split("\n"), df.tokens())
            for word in row_vocab.split("\n"):
                self.assertGreater(df[word], 0)
            with open(os.path.join(tmpdir, "shard-000-000.pb"), "rb") as fin:
                features = tf.parse_single_example(
                    fin.read(),
                    features={
                        "global_row": tf.FixedLenFeature([self.VOCAB], dtype=tf.int64),
                        "global_col": tf.FixedLenFeature([self.VOCAB], dtype=tf.int64),
                        "sparse_local_row": tf.VarLenFeature(dtype=tf.int64),
                        "sparse_local_col": tf.VarLenFeature(dtype=tf.int64),
                        "sparse_value": tf.VarLenFeature(dtype=tf.float32)
                    })
            with tf.Session() as session:
                global_row, global_col, local_row, local_col, value = session.run(
                    [features[n] for n in ("global_row", "global_col", "sparse_local_row",
                                           "sparse_local_col", "sparse_value")])
            self.assertEqual(set(range(self.VOCAB)), set(global_row))
            self.assertEqual(set(range(self.VOCAB)), set(global_col))
            nnz = 1421193
            self.assertEqual(value.values.shape, (nnz,))
            self.assertEqual(local_row.values.shape, (nnz,))
            self.assertEqual(local_col.values.shape, (nnz,))
            numpy.random.seed(0)
            all_tokens = row_vocab.split("\n")
            chosen_indices = numpy.random.choice(list(range(self.VOCAB)), 128, replace=False)
            chosen = [all_tokens[i] for i in chosen_indices]
            freqs = numpy.zeros((len(chosen),) * 2, dtype=int)
            index = {w: i for i, w in enumerate(chosen)}
            chosen = set(chosen)
            for path in os.listdir(args.input[0]):
                with asdf.open(os.path.join(args.input[0], path)) as model:
                    if model.tree["meta"]["model"] != "co-occurrences":
                        continue
                    matrix = assemble_sparse_matrix(model.tree["matrix"]).tocsr()
                    tokens = split_strings(model.tree["tokens"])
                    interesting = {i for i, t in enumerate(tokens) if t in chosen}
                    for y in interesting:
                        row = matrix[y]
                        yi = index[tokens[y]]
                        for x, v in zip(row.indices, row.data):
                            if x in interesting:
                                freqs[yi, index[tokens[x]]] += v
            matrix = coo_matrix((
                value.values, ([global_row[row] for row in local_row.values],
                               [global_col[col] for col in local_col.values])),
                shape=(self.VOCAB, self.VOCAB))
            matrix = matrix.tocsr()[chosen_indices][:, chosen_indices].todense().astype(int)
            self.assertTrue((matrix == freqs).all())

    def test_swivel_bad_params_submatrix_cols(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = default_swivel_args(tmpdir)
            args.submatrix_cols = 4097
            self.assertRaises(ValueError, lambda: run_swivel(args))

            args.submatrix_cols = 4096
            args.submatrix_rows = 4097
            self.assertRaises(ValueError, lambda: run_swivel(args))

    def test_swivel(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = default_swivel_args(tmpdir)
            run_swivel(args)
            check_swivel_results(self, tmpdir)

    def test_swivel_transformer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sw = SwivelTransformer()
            args = dict()
            args["input_base_path"] = os.path.join(os.path.dirname(__file__), "swivel")
            prepare_shard(args["input_base_path"])
            args["output_base_path"] = tmpdir
            args["embedding_size"] = 50
            args["num_epochs"] = 20
            sw.transform(**args)
            check_swivel_results(self, tmpdir)

    def test_swivel_transformer_logs(self):
        self.assertTrue(SwivelTransformer()._get_log_name())

    def test_postproc(self):
        with tempfile.NamedTemporaryFile(suffix=".asdf") as tmp:
            args = argparse.Namespace(
                swivel_output_directory=os.path.join(os.path.dirname(__file__), "postproc"),
                result=tmp.name)
            prepare_postproc_files(args.swivel_output_directory)

            postprocess(args)

            check_postproc_results(self, tmp.name)

    def test_postproc_transformer(self):
        with tempfile.NamedTemporaryFile(suffix=".asdf") as tmp:
            args = dict()
            args["swivel_output_directory"] = os.path.join(os.path.dirname(__file__), "postproc")
            args["result"] = tmp.name
            prepare_postproc_files(args["swivel_output_directory"])

            postproc = PostprocessTransformer()
            postproc.transform(**args)

            check_postproc_results(self, tmp.name)

    def test_postproc_transformer_logs(self):
        self.assertTrue(PostprocessTransformer()._get_log_name())


if __name__ == "__main__":
    unittest.main()
