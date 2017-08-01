import argparse
import os
import tempfile
import unittest

import asdf

import ast2vec.tests as tests
from ast2vec import Repo2nBOW, Id2Vec, DocumentFrequencies, Repo2nBOWTransformer
from ast2vec.repo2.nbow import repo2nbow_entry
from ast2vec.tests.models import ID2VEC, DOCFREQ


def validate_asdf_file(obj, filename):
    data = asdf.open(filename)
    obj.assertIn("meta", data.tree)
    obj.assertIn("repos", data.tree)
    obj.assertIn("matrix", data.tree)
    obj.assertEqual(2, len(data.tree["meta"]["dependencies"]))


class Repo2NBOWTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tests.setup()

    def test_obj(self):
        basedir = os.path.dirname(__file__)
        id2vec = Id2Vec().load(os.path.join(basedir, ID2VEC))
        df = DocumentFrequencies().load(os.path.join(basedir, DOCFREQ))
        df._df["xxyyzz"] = 10
        id2vec._token2index[id2vec.tokens[0]] = 1
        id2vec.tokens[0] = "xxyyzz"
        id2vec._token2index["xxyyzz"] = 0
        xxyyzz = Repo2nBOW(
            bblfsh_endpoint=os.getenv("BBLFSH_ENDPOINT", "0.0.0.0:9432"),
            id2vec=id2vec, docfreq=df, linguist=tests.ENRY, timeout=600)
        nbow = xxyyzz.convert_repository(os.path.join(basedir, "..", ".."))
        self.assertIsInstance(nbow, dict)
        self.assertAlmostEqual(nbow[0], 5.059296557734522)

    def test_asdf(self):
        basedir = os.path.dirname(__file__)
        id2vec = Id2Vec().load(os.path.join(basedir, ID2VEC))
        del id2vec._token2index[id2vec.tokens[0]]
        id2vec.tokens[0] = "test"
        id2vec._token2index["test"] = 0
        df = DocumentFrequencies().load(os.path.join(basedir, DOCFREQ))
        df._df["test"] = 10
        with tempfile.NamedTemporaryFile() as file:
            args = argparse.Namespace(
                id2vec=id2vec, docfreq=df, linguist=tests.ENRY,
                gcs_bucket=None, output=file.name,
                bblfsh_endpoint=os.getenv("BBLFSH_ENDPOINT", "0.0.0.0:9432"), timeout=None,
                repository=os.path.join(basedir, "..", ".."))
            repo2nbow_entry(args)
            self.assertTrue(os.path.isfile(file.name))
            validate_asdf_file(self, file.name)


class Repo2NBOWTransformerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tests.setup()

    def test_transform(self):
        basedir = os.path.dirname(__file__)
        id2vec = Id2Vec().load(os.path.join(basedir, ID2VEC))
        del id2vec._token2index[id2vec.tokens[0]]
        id2vec.tokens[0] = "test"
        id2vec._token2index["test"] = 0
        df = DocumentFrequencies().load(os.path.join(basedir, DOCFREQ))
        df._df["test"] = 10
        with tempfile.TemporaryDirectory() as tmpdir:
            repo2nbow = Repo2nBOWTransformer(
                id2vec=id2vec, docfreq=df, linguist=tests.ENRY, gcs_bucket=None,
                bblfsh_endpoint=os.getenv("BBLFSH_ENDPOINT", "0.0.0.0:9432"),
                timeout=None
            )
            outfile = repo2nbow.prepare_filename(basedir, tmpdir)
            repo2nbow.transform(repos=basedir, output=tmpdir)
            self.assertTrue(os.path.isfile(outfile))
            validate_asdf_file(self, outfile)

    def test_empty(self):
        basedir = os.path.dirname(__file__)
        with tempfile.TemporaryDirectory() as tmpdir:
            repo2nbow = Repo2nBOWTransformer(
                id2vec=os.path.join(basedir, ID2VEC),
                docfreq=os.path.join(basedir, DOCFREQ), linguist=tests.ENRY,
                gcs_bucket=None,
                bblfsh_endpoint=os.getenv("BBLFSH_ENDPOINT", "0.0.0.0:9432"),
                timeout=None
            )
            repo2nbow.transform(repos=os.path.join(basedir, "coocc"), output=tmpdir)
            self.assertFalse(os.listdir(tmpdir))

if __name__ == "__main__":
    unittest.main()
