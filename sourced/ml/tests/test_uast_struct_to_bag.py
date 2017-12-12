import unittest

from bblfsh import BblfshClient

from sourced.ml.algorithms import UastRandomWalk2Bag, UastSeq2Bag
from sourced.ml.tests.models import SOURCE_PY


class Uast2RandomWalk2BagTest(unittest.TestCase):
    def setUp(self):
        self.bag_extractor = UastRandomWalk2Bag(seq_len=[2, 3])
        self.uast = BblfshClient("0.0.0.0:9432").parse(SOURCE_PY).uast

    def test_uast_to_bag(self):
        bag = self.bag_extractor.uast_to_bag(self.uast)
        self.assertTrue(len(bag) > 0, "Expected size of bag should be > 0")


class UastSeq2BagTest(unittest.TestCase):
    def setUp(self):
        self.bag_extractor = UastSeq2Bag(seq_len=[2, 3])
        self.uast = BblfshClient("0.0.0.0:9432").parse(SOURCE_PY).uast

    def test_uast_to_bag(self):
        bag = self.bag_extractor.uast_to_bag(self.uast)
        self.assertTrue(len(bag) > 0, "Expected size of bag should be > 0")


if __name__ == "__main__":
    unittest.main()
