import os
import tempfile
import unittest

from ast2vec.model2.base import Model2Base


class FromModel:
    NAME = "from"

    def load(self, source):
        pass


class ToModel:
    NAME = "to"
    output = None

    def save(self, output, deps=None):
        ToModel.output = output


class Model2Test(Model2Base):
    MODEL_FROM_CLASS = FromModel
    MODEL_TO_CLASS = ToModel
    finalized = False

    def convert_model(self, model):
        return ToModel()


class MockingModel2Test(Model2Base):
    MODEL_FROM_CLASS = FromModel
    MODEL_TO_CLASS = ToModel
    finalized = False

    def convert_model(self, model):
        return ToModel()

    def finalize(self, index: int, destdir: str):
        self.finalized = True


class RaisingModel2Test(Model2Base):
    MODEL_FROM_CLASS = FromModel
    MODEL_TO_CLASS = ToModel

    def convert_model(self, model):
        raise ValueError("happens")


class FakeQueue:
    def __init__(self, contents: list):
        self.contents = contents

    def get(self):
        return self.contents.pop()

    def put(self, item):
        self.contents.append(item)


class Model2BaseTests(unittest.TestCase):
    def test_convert(self):
        converter = Model2Test(num_processes=2)
        status = converter.convert(os.path.dirname(__file__), "xxx", pattern="**/*.py")
        self.assertGreater(status, 20)

    def test_process_entry(self):
        converter = MockingModel2Test(num_processes=2)
        queue_in = FakeQueue([None, "srcdir/job"])
        queue_out = FakeQueue([])
        with tempfile.TemporaryDirectory(prefix="ast2vec-") as tmpdir:
            converter._process_entry(
                0, os.path.join(tmpdir, "destdir"), "srcdir", queue_in, queue_out)
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "destdir")))
            self.assertEqual(ToModel.output, os.path.join(tmpdir, "destdir", "job"))
        self.assertTrue(converter.finalized)
        self.assertEqual(queue_out.contents, [("srcdir/job", True)])

    def test_process_entry_exception(self):
        converter = RaisingModel2Test(num_processes=2)
        queue_in = FakeQueue([None, "srcdir/job"])
        queue_out = FakeQueue([])
        converter._process_entry(0, "destdir", "srcdir", queue_in, queue_out)
        self.assertEqual(queue_out.contents, [("srcdir/job", False)])


if __name__ == "__main__":
    unittest.main()
