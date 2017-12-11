import argparse
import os
import subprocess
import tempfile
import unittest

from sourced.ml.utils import install_bigartm


class BigartmTests(unittest.TestCase):
    gitdir = os.path.join(os.path.dirname(__file__), "..", "..")

    @unittest.skipUnless(os.getenv("FULL_TEST", False), "Need to define FULL_TEST env var.")
    def test_install_bigartm(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = argparse.Namespace(output=tmpdir, tmpdir=None)
            self.assertIsNone(install_bigartm(args))
            self._valivate_bigartm(tmpdir)

    def _valivate_bigartm(self, tmpdir):
        bigartm = os.path.join(tmpdir, "bigartm")
        self.assertTrue(os.path.isfile(bigartm))
        self.assertEqual(os.stat(bigartm).st_mode & 0o777, 0o777)
        output = subprocess.check_output([bigartm], stderr=subprocess.STDOUT)
        self.assertIn("BigARTM v", output.decode())


if __name__ == "__main__":
    unittest.main()
