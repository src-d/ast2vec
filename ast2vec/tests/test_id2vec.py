import os
import unittest

import numpy

from ast2vec import Id2Vec
import ast2vec.tests.models as paths


class Id2VecTests(unittest.TestCase):
    def setUp(self):
        self.model = Id2Vec().load(
            source=os.path.join(os.path.dirname(__file__), paths.ID2VEC))

    def test_embeddings(self):
        embeddings = self.model.embeddings
        self.assertIsInstance(embeddings, numpy.ndarray)
        self.assertEqual(embeddings.shape, (1000, 300))

    def test_tokens(self):
        tokens = self.model.tokens
        self.assertIsInstance(tokens, list)
        self.assertEqual(len(tokens), 1000)
        self.assertIsInstance(tokens[0], str)

    def test_token2index(self):
        self.assertEqual(self.model["get"], 0)
        with self.assertRaises(KeyError):
            print(self.model["xxx"])


if __name__ == "__main__":
    unittest.main()
