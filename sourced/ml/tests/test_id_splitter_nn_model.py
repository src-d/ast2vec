import string
import tempfile
import unittest

import numpy

from sourced.ml.tests import has_tensorflow
from sourced.ml.tests.models import ID_SPLITTER_RNN


class MetricsTests(unittest.TestCase):

    @unittest.skipIf(not has_tensorflow(), "Tensorflow is not installed.")
    def test_register_metric(self):
        from sourced.ml.algorithms.id_splitter.nn_model import register_metric, METRICS
        fake_metric = "fake metric"
        register_metric(fake_metric)
        self.assertIn(fake_metric, METRICS)
        METRICS.pop()
        self.assertNotIn(fake_metric, METRICS)

    @unittest.skipIf(not has_tensorflow(), "Tensorflow is not installed.")
    def test_raise_register_metric(self):
        from sourced.ml.algorithms.id_splitter.nn_model import register_metric, METRICS
        bad_metric = 1
        with self.assertRaises(AssertionError):
            register_metric(bad_metric)
        self.assertNotIn(bad_metric, METRICS)


class ModelsTests(unittest.TestCase):
    def setUp(self):
        from sourced.ml.algorithms.id_splitter.nn_model import build_rnn, build_cnn
        self.n_uniq = len(string.ascii_lowercase)
        self.model_rnn = build_rnn(maxlen=5, units=24, stack=2, rnn_layer="LSTM",
                                   optimizer="Adam", dev0="/cpu:0", dev1="/cpu:0")
        self.model_cnn = build_cnn(maxlen=5, filters=[64, 32, 16, 8], output_n_filters=32,
                                   stack=2, kernel_sizes=[2, 4, 8, 16], optimizer="Adam",
                                   device="/cpu:0")

    @unittest.skipIf(not has_tensorflow(), "Tensorflow is not installed.")
    def test_build_rnn(self):
        self.assertTrue(self.model_rnn.built)
        self.assertTrue(self.model_rnn.trainable)
        self.assertIsInstance(self.model_rnn.get_weights()[0], numpy.ndarray)
        self.assertEqual(self.model_rnn.get_weights()[0].shape, (self.n_uniq+1, self.n_uniq+1))
        self.assertTrue(self.model_rnn.uses_learning_phase)

    @unittest.skipIf(not has_tensorflow(), "Tensorflow is not installed.")
    def test_build_cnn(self):
        self.assertTrue(self.model_cnn.built)
        self.assertTrue(self.model_cnn.trainable)
        self.assertIsInstance(self.model_cnn.get_weights()[0], numpy.ndarray)
        self.assertEqual(self.model_cnn.get_weights()[0].shape, (self.n_uniq+1, self.n_uniq+1))
        self.assertTrue(self.model_cnn.uses_learning_phase)


class NNModelTest(unittest.TestCase):

    def setUp(self):
        from sourced.ml.models.id_splitter import IdentifierSplitterNN
        self.test_X = ["networkSocket", "variablename", "loadfile", "blahblah", "foobar"]
        self.test_y = ["network", "socket", "variable",
                       "name", "load", "file", "blah", "blah", "foobar"]
        self.id_splitter = IdentifierSplitterNN()
        self.id_splitter.load(ID_SPLITTER_RNN)

    @unittest.skipIf(not has_tensorflow(), "Tensorflow is not installed.")
    def test_load_and_run_model(self):
        self.assertEqual(self.id_splitter(self.test_X), self.test_y)

    @unittest.skipIf(not has_tensorflow(), "Tensorflow is not installed.")
    def test_save_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.id_splitter.save(tmpdir + "/model.asdf", series="id-splitter-nn")
            self.id_splitter.load(tmpdir + "/model.asdf")
        self.assertEqual(self.id_splitter(self.test_X), self.test_y)


if __name__ == "__main__":
    unittest.main()
