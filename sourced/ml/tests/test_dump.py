import argparse
import logging
import os
import sys
import unittest
from contextlib import contextmanager
from io import StringIO

import modelforge.gcs_backend as gcs_backend
import sourced.ml.tests.models as paths
from sourced.ml.tests.fake_requests import FakeRequests
from sourced.ml.cmd_entries import dump_model


@contextmanager
def captured_output():
    log = StringIO()
    log_handler = logging.StreamHandler(log)
    logging.getLogger().addHandler(log_handler)
    new_out, new_err = StringIO(), StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr, log
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        logging.getLogger().removeHandler(log_handler)


class DumpTests(unittest.TestCase):
    ID2VEC_DUMP = """{'created_at': datetime.datetime(2017, 6, 18, 17, 37, 6, 255615),
 'dependencies': [],
 'model': 'id2vec',
 'uuid': '92609e70-f79c-46b5-8419-55726e873cfc',
 'version': [1, 0, 0]}
Shape: (1000, 300)
First 10 words: ['get', 'name', 'type', 'string', 'class', 'set', 'data', 'value', 'self', 'test']
"""
    DOCFREQ_DUMP = """{'created_at': datetime.datetime(2017, 8, 9, 16, 49, 12, 775367),
 'dependencies': [],
 'model': 'docfreq',
 'uuid': 'f64bacd4-67fb-4c64-8382-399a8e7db52a',
 'version': [0, 1, 0]}
Number of words: 982
""" + "Random 10 words: "

    BOW_DUMP = """{'created_at': datetime.datetime(2018, 1, 18, 21, 59, 59, 200818),
 'dependencies': [{'created_at': datetime.datetime(2018, 1, 18, 21, 59, 48, 828287),
                   'dependencies': [],
                   'model': 'docfreq',
                   'uuid': '2c4fcae7-93a6-496e-9e3a-d6e15d35b812',
                   'version': [1, 0, 0]}],
 'model': 'bow',
 'parent': '51b4165d-b2c6-442a-93be-0eb35f4cc19a',
 'uuid': '0d95f342-2c69-459f-9ee7-a1fc7da88d64',
 'version': [1, 0, 15]}
Shape: (5, 20)
First 10 documents: ['repo1', 'repo2', 'repo3', 'repo4', 'repo5']
First 10 tokens: ['i.', 'i.*', 'i.Activity', 'i.AdapterView', 'i.ArrayAdapter', 'i.Arrays', 'i.Bundle', 'i.EditText', 'i.Exception', 'i.False']\n"""  # nopep8

    COOCC_DUMP = """{'created_at': datetime.datetime(2018, 1, 24, 16, 0, 2, 591553),
 'dependencies': [{'created_at': datetime.datetime(2018, 1, 24, 15, 59, 24, 129470),
                   'dependencies': [],
                   'model': 'docfreq',
                   'uuid': '0f94a6c6-7dc3-4b3c-b8d2-917164a50581',
                   'version': [1, 0, 0]}],
 'model': 'co-occurrences',
 'uuid': 'e75dcb2d-ec1d-476b-a04b-bc64c7779ae1',
 'version': [1, 0, 0]}
Number of words: 304
First 10 words: ['i.set', 'i.iter', 'i.error', 'i.logsdir', 'i.read', 'i.captur', 'i.clear',""" + \
                 """ 'i.android', 'i.tohome', 'i.ljust']
Matrix: shape: (304, 304) non-zero: 16001
"""

    def test_id2vec(self):
        with captured_output() as (out, _, _):
            dump_model(self._get_args(input=paths.ID2VEC))
        self.assertEqual(out.getvalue(), self.ID2VEC_DUMP)

    def test_docfreq(self):
        with captured_output() as (out, _, _):
            dump_model(self._get_args(input=paths.DOCFREQ))
        self.assertEqual(out.getvalue()[:len(self.DOCFREQ_DUMP)], self.DOCFREQ_DUMP)
        ending = "\nNumber of documents: 1000\n"
        self.assertEqual(out.getvalue()[-len(ending):], ending)

    def test_bow(self):
        with captured_output() as (out, _, _):
            dump_model(self._get_args(input=paths.BOW))
        self.assertEqual(out.getvalue(), self.BOW_DUMP)

    def test_coocc(self):
        with captured_output() as (out, _, _):
            dump_model(self._get_args(input=paths.COOCC))
        self.assertEqual(out.getvalue(), self.COOCC_DUMP)

    def test_id2vec_id(self):
        def route(url):
            if gcs_backend.INDEX_FILE in url:
                return '{"models": {"id2vec": {' \
                       '"92609e70-f79c-46b5-8419-55726e873cfc": ' \
                       '{"url": "https://xxx"}}}}'.encode()
            self.assertEqual("https://xxx", url)
            with open(paths.ID2VEC, "rb") as fin:
                return fin.read()

        gcs_backend.requests = FakeRequests(route)
        with captured_output() as (out, err, _):
            dump_model(self._get_args(
                input="92609e70-f79c-46b5-8419-55726e873cfc"))
        self.assertEqual(out.getvalue(), self.ID2VEC_DUMP)
        self.assertFalse(err.getvalue())

    def test_id2vec_url(self):
        def route(url):
            self.assertEqual("https://xxx", url)
            with open(paths.ID2VEC, "rb") as fin:
                return fin.read()

        gcs_backend.requests = FakeRequests(route)
        with captured_output() as (out, _, _):
            dump_model(self._get_args(input="https://xxx"))
        self.assertEqual(out.getvalue(), self.ID2VEC_DUMP)

    def test_gcs(self):
        def route(url):
            if gcs_backend.INDEX_FILE in url:
                self.assertIn("custom", url)
                return '{"models": {"id2vec": {' \
                       '"92609e70-f79c-46b5-8419-55726e873cfc": ' \
                       '{"url": "https://xxx"}}}}'.encode()
            self.assertEqual("https://xxx", url)
            with open(paths.ID2VEC, "rb") as fin:
                return fin.read()

        gcs_backend.requests = FakeRequests(route)
        with captured_output() as (out, _, _):
            dump_model(self._get_args(input="92609e70-f79c-46b5-8419-55726e873cfc",
                                      gcs="custom"))
        self.assertEqual(out.getvalue(), self.ID2VEC_DUMP)

    @staticmethod
    def _get_args(input=None, gcs=None, dependency=tuple()):
        return argparse.Namespace(input=input, gcs_bucket=gcs, dependency=dependency,
                                  log_level="WARNING")


if __name__ == "__main__":
    unittest.main()
