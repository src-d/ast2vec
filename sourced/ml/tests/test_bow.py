import os
import unittest

import numpy

import sourced.ml.tests.models as paths
from sourced.ml.models import NBOW, BOW


class NBOWTests(unittest.TestCase):
    def setUp(self):
        self.model = NBOW().load(
            source=os.path.join(os.path.dirname(__file__), paths.NBOW))

    def test_getitem(self):
        repo_name, indices, weights = self.model[0]
        self.assertEqual(repo_name, "ikizir/HohhaDynamicXOR")
        self.assertIsInstance(indices, numpy.ndarray)
        self.assertIsInstance(weights, numpy.ndarray)
        self.assertEqual(indices.shape, weights.shape)
        self.assertEqual(indices.shape, (85,))

    def test_iter(self):
        pumped = list(self.model)
        self.assertEqual(len(pumped), 1000)
        self.assertEqual(pumped, list(range(1000)))

    def test_len(self):
        self.assertEqual(len(self.model), 1000)

    def test_repository_index_by_name(self):
        self.assertEqual(
            self.model.repository_index_by_name("ikizir/HohhaDynamicXOR"), 0)


class BOWTests(unittest.TestCase):
    def setUp(self):
        self.model = BOW().load(
            source=os.path.join(os.path.dirname(__file__), paths.BOW))

    def test_getitem(self):
        repo_name, indices, weights = self.model[0]
        self.assertEqual(repo_name, "ikizir/HohhaDynamicXOR")
        self.assertIsInstance(indices, numpy.ndarray)
        self.assertIsInstance(weights, numpy.ndarray)
        self.assertEqual(indices.shape, weights.shape)
        self.assertEqual(indices.shape, (85,))

    def test_iter(self):
        pumped = list(self.model)
        self.assertEqual(len(pumped), 1000)
        self.assertEqual(pumped, list(range(1000)))

    def test_len(self):
        self.assertEqual(len(self.model), 1000)

    def test_repository_index_by_name(self):
        self.assertEqual(
            self.model.repository_index_by_name("ikizir/HohhaDynamicXOR"), 0)

    def test_tokens(self):
        self.assertEqual(self.model.tokens[0], "get")


if __name__ == "__main__":
    unittest.main()
