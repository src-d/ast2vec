import argparse
import os
import tempfile
import unittest

import sourced.ml.tests.models as paths
from sourced.ml.models import Topics
from sourced.ml.utils import bigartm2asdf_entry


class TopicsTests(unittest.TestCase):
    def setUp(self):
        self.model = Topics().load(os.path.join(os.path.dirname(__file__), paths.TOPICS))

    def test_dump(self):
        res = self.model.dump()
        self.assertEqual(res, """320 topics, 1000 tokens
First 10 tokens: ['ulcancel', 'domainlin', 'trudi', 'fncreateinstancedbaselin', 'wbnz', 'lmultiplicand', 'otronumero', 'qxln', 'gvgq', 'polaroidish']
Topics: unlabeled
non-zero elements: 6211  (0.019409)""")  # nopep8

    def test_props(self):
        self.assertEqual(len(self.model), 320)
        self.assertEqual(len(self.model.tokens), 1000)
        self.assertIsNone(self.model.topics)
        zt = self.model[0]
        self.assertEqual(len(zt), 8)
        self.assertEqual(zt[0][0], "olcustom")
        self.assertAlmostEqual(zt[0][1], 1.23752e-06, 6)

    def test_label(self):
        with self.assertRaises(ValueError):
            self.model.label_topics([1, 2, 3])
        with self.assertRaises(TypeError):
            self.model.label_topics(list(range(320)))
        self.model.label_topics([str(i) for i in range(320)])
        self.assertEqual(self.model.topics[0], "0")

    def test_save(self):
        with tempfile.NamedTemporaryFile(prefix="sourced.ml-topics-test-") as f:
            self.model.save(f.name)
            new = Topics().load(f.name)
            self.assertEqual(self.model.tokens, new.tokens)
            self.assertEqual((self.model.matrix != new.matrix).getnnz(), 0)

    def test_bigartm2asdf_entry(self):
        with tempfile.NamedTemporaryFile(prefix="sourced.ml-topics-test-") as f:
            args = argparse.Namespace(
                input=os.path.join(os.path.dirname(__file__), paths.TOPICS_SRC),
                output=f.name)
            bigartm2asdf_entry(args)
            model = Topics().load(f.name)
            self.assertEqual(len(model), 320)
            self.assertEqual(len(model.tokens), 1000)


if __name__ == "__main__":
    unittest.main()
