import argparse
import logging
import os
import tempfile
import unittest

from ast2vec.bow import NBOW, BOW
import ast2vec.vw_dataset
from ast2vec.vw_dataset import convert_bow_to_vw, bow2vw_entry
import ast2vec.tests.models as paths


class Bow2vwTests(unittest.TestCase):
    def test_convert_bow_to_vw(self):
        bow = NBOW().load(source=os.path.join(os.path.dirname(__file__), paths.NBOW))
        vocabulary = ["get", "name", "type", "string", "class", "set", "data", "value", "self",
                      "test"]
        bow.become_bow(vocabulary)
        with tempfile.NamedTemporaryFile(prefix="ast2vec-vw-") as fout:
            logging.getLogger().level = logging.ERROR
            try:
                convert_bow_to_vw(bow, fout.name)
            finally:
                logging.getLogger().level = logging.INFO
            fout.seek(0)
            contents = fout.read().decode()
        hits = 0
        for word in vocabulary:
            if " %s:" % word in contents:
                hits += 1
        self.assertEqual(hits, 6)

    def test_repo2bow_entry(self):
        called = [None] * 3

        def fake_convert_bow_to_vw(*args):
            called[:] = args

        args = argparse.Namespace(nbow=os.path.join(os.path.dirname(__file__), paths.NBOW),
                                  id2vec=os.path.join(os.path.dirname(__file__), paths.ID2VEC),
                                  output="out.test")
        backup = ast2vec.vw_dataset.convert_bow_to_vw
        ast2vec.vw_dataset.convert_bow_to_vw = fake_convert_bow_to_vw
        try:
            bow2vw_entry(args)
        finally:
            ast2vec.vw_dataset.convert_bow_to_vw = backup
        self.assertIsInstance(called[0], BOW)
        self.assertEqual(called[1], "out.test")


if __name__ == "__main__":
    unittest.main()
