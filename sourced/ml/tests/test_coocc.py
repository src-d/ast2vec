import os
import unittest

import sourced.ml.tests.models as paths
from sourced.ml.models import Cooccurrences


class CooccurrencesTests(unittest.TestCase):
    def setUp(self):
        self.model = Cooccurrences().load(
            source=os.path.join(os.path.dirname(__file__), paths.COOCC))

    def test_tokens(self):
        tokens = self.model.tokens
        self.assertIsInstance(tokens, list)
        self.assertEqual(tokens[:10], [
            "generic", "model", "dump", "printer", "pprint", "print", "nbow", "vec", "idvec",
            "coocc"])
        self.assertEqual(len(tokens), 394)

    def test_matrix(self):
        matrix = self.model.matrix
        self.assertEqual(matrix.shape, (394, 394))
        self.assertEqual(matrix.getnnz(), 20832)

    def test_len(self):
        self.assertEqual(len(self.model), 394)


if __name__ == "__main__":
    unittest.main()
