import os
import unittest

from ast2vec import DocumentFrequencies
import ast2vec.tests.models as paths


class DocumentFrequenciesTests(unittest.TestCase):
    def setUp(self):
        self.model = DocumentFrequencies().load(
            source=os.path.join(os.path.dirname(__file__), paths.DOCFREQ))

    def test_docs(self):
        docs = self.model.docs
        self.assertIsInstance(docs, int)
        self.assertEqual(docs, 1000)

    def test_get(self):
        self.assertEqual(self.model["aaaaaaa"], 341)
        with self.assertRaises(KeyError):
            print(self.model["xaaaaaa"])
        self.assertEqual(self.model.get("aaaaaaa", 0), 341)
        self.assertEqual(self.model.get("xaaaaaa", 100500), 100500)

    def test_tokens(self):
        tokens = self.model.tokens()
        self.assertEqual(sorted(tokens), tokens)
        for t in tokens:
            self.assertGreater(self.model[t], 0)

    def test_len(self):
        # the remaining 20 are not unique - the model was generated badly
        self.assertEqual(len(self.model), 980)


if __name__ == "__main__":
    unittest.main()
