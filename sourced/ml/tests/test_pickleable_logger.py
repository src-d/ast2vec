import logging
import pickle
import unittest

from sourced.ml.utils import PickleableLogger


class TestLogger(PickleableLogger):
    def _get_log_name(self):
        return "test"


class PickleableLoggerTests(unittest.TestCase):
    def test_pickle(self):
        logger = TestLogger(log_level=logging.ERROR)
        logger = pickle._loads(pickle._dumps(logger))
        self.assertIsInstance(logger._log, logging.Logger)
        self.assertEqual(logger._log.level, logging.ERROR)


if __name__ == "__main__":
    unittest.main()
