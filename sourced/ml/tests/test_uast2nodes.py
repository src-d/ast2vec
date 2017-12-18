import unittest

from bblfsh import BblfshClient

from sourced.ml.algorithms import UastNode2Bag, Quantization
from sourced.ml.tests.models import SOURCE_PY


class UastNode2BagTest(unittest.TestCase):
    def setUp(self):
        self.bag_extractor = UastNode2Bag(children2ind=Quantization())
        self.uast = BblfshClient("0.0.0.0:9432").parse(SOURCE_PY).uast

    def test_uast_to_bag(self):
        bag_ic = self.bag_extractor.uast_to_bag(self.uast, feature="internal_type")
        bag_rc = self.bag_extractor.uast_to_bag(self.uast, feature="roles")
        self.assertTrue(len(bag_ic) > 0, "Expected size of bag should be > 0")
        self.assertTrue(len(bag_rc) > 0, "Expected size of bag should be > 0")


if __name__ == "__main__":
    unittest.main()
