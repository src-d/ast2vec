import argparse
import os
import tempfile
import unittest

import asdf
from scipy.sparse import coo_matrix

from ast2vec import Repo2CooccTransformer, Repo2ProxSwivel
from ast2vec.repo2.prox import repo2prox_entry
import ast2vec.tests as tests


def validate_asdf_file(obj, filename):
    data = asdf.open(filename)
    obj.assertIn("meta", data.tree)
    obj.assertIn("matrix", data.tree)
    obj.assertIn("tokens", data.tree)
    obj.assertEqual(data.tree["meta"]["model"], "co-occurrences")


class Repo2ProxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tests.setup()

    def test_obj(self):
        basedir = os.path.dirname(__file__)
        repo2 = Repo2ProxSwivel(
            bblfsh_endpoint=os.getenv("BBLFSH_ENDPOINT", "0.0.0.0:9432"),
            linguist=tests.ENRY, timeout=600)
        prox = repo2.convert_repository(os.path.join(basedir, "..", ".."))
        self.assertIsInstance(prox, tuple)
        self.assertEqual(len(prox), 2)
        self.assertIn("RoleId_1", prox[0])
        self.assertIn("document", prox[0])
        self.assertIsInstance(prox[1], coo_matrix)
        self.assertEqual(prox[1].shape, (len(prox[0]),) * 2)

    def test_asdf(self):
        basedir = os.path.dirname(__file__)
        with tempfile.NamedTemporaryFile() as file:
            args = argparse.Namespace(
                linguist=tests.ENRY, output=file.name, matrix_type="Swivel",
                bblfsh_endpoint=os.getenv("BBLFSH_ENDPOINT", "0.0.0.0:9432"),
                timeout=None, repository=os.path.join(basedir, "..", ".."))
            repo2prox_entry(args)
            validate_asdf_file(self, file.name)

    def test_linguist(self):
        with self.assertRaises(FileNotFoundError):
            Repo2ProxSwivel(linguist="xxx", timeout=600)
        with self.assertRaises(FileNotFoundError):
            Repo2ProxSwivel(linguist=__file__, timeout=600)


class Repo2ProxTransformerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tests.setup()

    def test_transform(self):
        basedir = os.path.dirname(__file__)
        with tempfile.TemporaryDirectory() as tmpdir:
            repo2 = Repo2CooccTransformer(
                bblfsh_endpoint=os.getenv("BBLFSH_ENDPOINT", "0.0.0.0:9432"),
                linguist=tests.ENRY, timeout=600)
            repo2.transform(repos=basedir, output=tmpdir)

            # check that output file exists
            outfile = repo2.prepare_filename(basedir, tmpdir)
            self.assertEqual(os.path.exists(outfile), 1)

            validate_asdf_file(self, outfile)

    def test_empty(self):
        basedir = os.path.dirname(__file__)
        with tempfile.TemporaryDirectory() as tmpdir:
            repo2 = Repo2CooccTransformer(
                bblfsh_endpoint=os.getenv("BBLFSH_ENDPOINT", "0.0.0.0:9432"),
                linguist=tests.ENRY, timeout=600)
            repo2.transform(repos=os.path.join(basedir, "coocc"), output=tmpdir)
            self.assertFalse(os.listdir(tmpdir))

if __name__ == "__main__":
    unittest.main()
