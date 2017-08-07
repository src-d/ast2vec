import argparse
import unittest
import tempfile

from ast2vec.bow import NBOW
from ast2vec.model2.join_bow import joinbow_entry
import ast2vec.tests.models as paths


class JoinBowTests(unittest.TestCase):
    def test_join_nbow(self):
        with tempfile.NamedTemporaryFile(prefix="ast2vec-test-join-nbow-", suffix=".asdf") as tmpf:
            args = argparse.Namespace(processes=1, nbow=True, bow=False, input=paths.JOINBOWS,
                                      output=tmpf.name, tmpdir=None, filter="**/nbow_github*.asdf")
            joinbow_entry(args)
            merged = NBOW().load(tmpf.name)
        self.assertEqual({
            "github.com/src-d/vecino", "github.com/src-d/ast2vec", "github.com/src-d/modelforge"},
            set(merged.repos))
        self.assertEqual(merged._matrix.shape, (3, 1000))
        self.assertEqual(merged._matrix.getnnz(), 432)


if __name__ == "__main__":
    unittest.main()
