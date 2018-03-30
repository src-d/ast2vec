from io import BytesIO
import unittest

from sourced.ml.models import DocumentFrequencies
import sourced.ml.tests.models as paths


class DocumentFrequenciesTests(unittest.TestCase):
    def setUp(self):
        self.model = DocumentFrequencies().load(source=paths.DOCFREQ)

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
        self.assertEqual(list(self.model._df), self.model.tokens())

    def test_len(self):
        # the remaining 18 are not unique - the model was generated badly
        self.assertEqual(len(self.model), 982)

    def test_iter(self):
        aaa = False
        for tok, freq in self.model:
            if "aaaaaaa" in tok:
                aaa = True
                int(freq)
                break
        self.assertTrue(aaa)

    def test_prune(self):
        pruned = self.model.prune(4)
        for tok, freq in pruned:
            self.assertGreaterEqual(freq, 4)
        self.assertEqual(len(pruned), 346)

    def test_prune_self(self):
        pruned = self.model.prune(1)
        self.assertIs(self.model, pruned)

    def test_greatest(self):
        pruned = self.model.greatest(100)
        freqs = [v for v in self.model._df.values()]
        freqs.sort(reverse=True)
        border = freqs[100]
        for v in pruned._df.values():
            self.assertGreaterEqual(v, border)
        df1 = pruned._df
        df2 = self.model.greatest(100)._df
        self.assertEqual(df1, df2)

    def test_write(self):
        buffer = BytesIO()
        self.model.save(buffer)
        buffer.seek(0)
        new_model = DocumentFrequencies().load(buffer)
        self.assertEqual(self.model._df, new_model._df)
        self.assertEqual(self.model.docs, new_model.docs)


if __name__ == "__main__":
    unittest.main()
