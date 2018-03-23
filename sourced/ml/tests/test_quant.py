from io import BytesIO
import unittest

import numpy

from sourced.ml.models import QuantizationLevels
import sourced.ml.tests.models as paths


class QuantizationLevelsTests(unittest.TestCase):
    def setUp(self):
        self.model = QuantizationLevels().load(source=paths.QUANTLEVELS)

    def test_levels(self):
        levels = self.model.levels
        self.assertIsInstance(levels, dict)
        self.assertEqual(len(levels), 1)
        self.assertIsInstance(levels["children"], dict)
        self.assertEqual(len(levels["children"]), 259)

    def test_len(self):
        self.assertEqual(len(self.model), 1)

    def test_write(self):
        levels = {"xxx": {"a": numpy.array([1, 2, 3]), "b": numpy.array([4, 5, 6]),
                          "c": numpy.array([7, 8, 9])},
                  "yyy": {"q": numpy.array([3, 2, 1]), "w": numpy.array([6, 5, 4]),
                          "e": numpy.array([9, 8, 7])}}
        buffer = BytesIO()
        QuantizationLevels().construct(levels).save(buffer)
        buffer.seek(0)
        model = QuantizationLevels().load(buffer)
        levels = model.levels
        self.assertEqual(len(levels), 2)
        self.assertEqual(len(levels["xxx"]), 3)
        self.assertEqual(len(levels["yyy"]), 3)
        self.assertTrue((levels["xxx"]["a"] == numpy.array([1, 2, 3])).all())
        self.assertTrue((levels["xxx"]["b"] == numpy.array([4, 5, 6])).all())
        self.assertTrue((levels["xxx"]["c"] == numpy.array([7, 8, 9])).all())
        self.assertTrue((levels["yyy"]["q"] == numpy.array([3, 2, 1])).all())
        self.assertTrue((levels["yyy"]["w"] == numpy.array([6, 5, 4])).all())
        self.assertTrue((levels["yyy"]["e"] == numpy.array([9, 8, 7])).all())

    def test_dump(self):
        self.assertEqual(self.model.dump(), "Schemes: [('children', '259@10')]")


if __name__ == "__main__":
    unittest.main()
