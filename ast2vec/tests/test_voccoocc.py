import os
import unittest

from ast2vec.voccoocc import VocabularyCooccurrences
import ast2vec.tests.models as paths


class CooccurrencesTests(unittest.TestCase):
    def setUp(self):
        self.model = VocabularyCooccurrences().load(
            source=os.path.join(os.path.dirname(__file__), paths.VOCCOOCC))

    def test_matrix(self):
        matrix = self.model.matrix
        self.assertEqual(matrix.shape, (394, 394))
        self.assertEqual(matrix.getnnz(), 20832)

    def test_len(self):
        self.assertEqual(len(self.model), 394)


if __name__ == "__main__":
    unittest.main()
