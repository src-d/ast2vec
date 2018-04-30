import os
import sys
import unittest

from pyspark import Row

from sourced.ml.tests import tfidf_data
from sourced.ml.transformers import ContentToIdentifiers, IdentifiersToDataset
from sourced.ml.utils import create_spark


class Content2IdsTests(unittest.TestCase):
    def setUp(self):
        self.sc = create_spark("test")

    def test_languages(self):
        for path in sys.path:
            path_to_languages = os.path.join(path, "sourced/ml/transformers/languages.yml")
            if os.path.exists(path_to_languages):
                break
        self.assertTrue(os.path.exists(path_to_languages))

    def test_call(self):
        content2ids = ContentToIdentifiers(split=False)
        ids2dataset = IdentifiersToDataset(idfreq=False)
        for data, result in zip(tfidf_data.datasets, tfidf_data.ids_result):
            df = self.sc.sparkContext \
                .parallelize(range(len(data["content"]))) \
                .map(lambda x: Row(content=data["content"][x], path=str(data["file"][x]),
                                   repository_id=str(data["document"][x]), lang=data["lang"][x])) \
                .toDF()
            rdd_processed = content2ids(df)
            self.assertEqual(result, set(ids2dataset(rdd_processed).collect()))

    def test_call_split_idfreq(self):
        content2ids = ContentToIdentifiers(split=True)
        ids2dataset = IdentifiersToDataset(idfreq=True)
        for data, result in zip(tfidf_data.datasets, tfidf_data.ids_split_idfreq_result):
            df = self.sc.sparkContext \
                .parallelize(range(len(data["content"]))) \
                .map(lambda x: Row(content=data["content"][x], path=str(data["file"][x]),
                                   repository_id=str(data["document"][x]), lang=data["lang"][x])) \
                .toDF()
            rdd_processed = content2ids(df)
            self.assertEqual(result, set(ids2dataset(rdd_processed).collect()))

    def test_process_row_split(self):
        content2ids = ContentToIdentifiers(split=True)
        row = Row(content="from foo import FooBar; print('foobar')",
                  path="foobar.py",
                  repository_id="src-d/ml",
                  lang="Python")
        row_md = Row(content="#title",
                     path="README",
                     repository_id="src-d/ml",
                     lang="Markdown")
        row_batch = Row(content="@ECHO OFF; ECHO Hello World!",
                        path="bat",
                        repository_id="src-d/ml",
                        lang="Batchfile")
        self.assertEqual(list(content2ids.process_row(row)),
                         [("FooBar", ("src-d/ml", "src-d/ml/foobar.py"))])
        self.assertEqual(len(list(content2ids.process_row(row_md))), 0)
        self.assertEqual(len(list(content2ids.process_row(row_batch))), 0)

    def test_process_row(self):
        content2ids = ContentToIdentifiers(split=False)
        row = Row(content="from foo import FooBar; print('foobar')",
                  path="foobar.py",
                  repository_id="src-d/ml",
                  lang="Python")
        self.assertEqual(list(content2ids.process_row(row)),
                         [("foo", ("src-d/ml", "src-d/ml/foobar.py")),
                          ("FooBar", ("src-d/ml", "src-d/ml/foobar.py"))])


if __name__ == "__main__":
    unittest.main()
