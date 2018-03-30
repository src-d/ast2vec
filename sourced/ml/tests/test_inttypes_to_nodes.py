import unittest

from bblfsh import BblfshClient

from sourced.ml.algorithms import Uast2QuantizedChildren
from sourced.ml.tests.models import SOURCE_PY


class Uast2NodesBagTest(unittest.TestCase):
    def setUp(self):
        self.nodes_bag_extractor = Uast2QuantizedChildren(npartitions=3)
        self.uast = BblfshClient("0.0.0.0:9432").parse(SOURCE_PY).uast

    def test_uast_to_bag(self):
        bag = self.nodes_bag_extractor(self.uast)
        self.assertTrue(len(bag) > 0, "Expected size of bag should be > 0")

    def test_quantize_1(self):
        freqs = {1: 100, 2: 90, 3: 10, 5: 10, 6: 5, 7: 5}
        levels = self.nodes_bag_extractor.quantize_unwrapped(freqs.items())
        self.assertEqual(list(levels), [1, 2, 3, 7])

    def test_quantize_2(self):
        freqs = {1: 10, 2: 10, 3: 10, 5: 10, 6: 10, 7: 10}
        levels = self.nodes_bag_extractor.quantize_unwrapped(freqs.items())
        self.assertEqual(list(levels), [1, 3, 6, 7])

    def test_quantize_3(self):
        freqs = {1: 100, 2: 1, 3: 1, 5: 1, 6: 1, 7: 1}
        levels = self.nodes_bag_extractor.quantize_unwrapped(freqs.items())
        self.assertEqual(list(levels), [1, 2, 7, 7])

    def test_quantize_4(self):
        freqs = {1: 10, 2: 15, 3: 5, 5: 15, 6: 10, 7: 10}
        levels = self.nodes_bag_extractor.quantize_unwrapped(freqs.items())
        self.assertEqual(list(levels), [1, 2, 5, 7])


if __name__ == "__main__":
    unittest.main()
