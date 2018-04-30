import unittest
import argparse

from sourced.ml.tests.models import PARQUET_DIR
from sourced.ml.transformers import create_uast_source


class LoaderTest(unittest.TestCase):
    def test_parquet(self):
        args = argparse.Namespace(parquet=True, repositories=PARQUET_DIR)
        root, start_point = create_uast_source(args, "test_parquet")
        self.assertEqual(root, start_point)
        df = start_point.execute()
        self.assertEqual(df.count(), 6)
        row = df.rdd.first()
        self.assertEqual(len(row), 5)
        self.assertEqual(row.blob_id, "c8f29225eed4e8a56b4085e3c1469f336e67f4c7")
        self.assertEqual(row.repository_id, "github.com/sloria/flask-ghpages-example")
        self.assertEqual(row.path, "freeze.py")
        self.assertEqual(row.commit_hash, "e08278f331b2450441f7879c57ad574d9caf2032")
