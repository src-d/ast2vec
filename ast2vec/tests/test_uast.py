import os
import unittest

from ast2vec.uast import UASTModel
import ast2vec.tests.models as paths


class UASTTests(unittest.TestCase):
    def setUp(self):
        self.model = UASTModel().load(
            source=os.path.join(os.path.dirname(__file__), paths.UAST))

    def test_filenames(self):
        filenames = self.model.filenames
        self.assertIsInstance(filenames, list)
        self.assertEqual(
            filenames,
            ["setup.py", "jgscm/tests/__init__.py", "jgscm/tests/test.py", "jgscm/__init__.py"])

    def test_uasts(self):
        uasts = self.model.uasts
        self.assertIsInstance(uasts, list)
        self.assertEqual(len(uasts), len(self.model.filenames))
        self.assertEqual(uasts[0].SerializeToString()[:11], b"\x1a\xa1\x1c\n\x06Module")

    def test_len(self):
        self.assertEqual(len(self.model), 4)


if __name__ == "__main__":
    unittest.main()
