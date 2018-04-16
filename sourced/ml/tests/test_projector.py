import json
import os
import shutil
import socket
import tempfile
import time
import unittest

import requests

from modelforge.logs import setup_logging
from sourced.ml.tests.test_dump import captured_output
from sourced.ml.utils.projector import CORSWebServer, web_server, wait, present_embeddings


class ProjectorTests(unittest.TestCase):
    MAX_ATTEMPTS = 40

    @classmethod
    def setUpClass(cls):
        setup_logging("DEBUG")

    def setUp(self):
        self.pwd = os.getcwd()

    def tearDown(self):
        os.chdir(self.pwd)

    def wait_for_web_server(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            result = -1
            attempts = 0
            while result != 0 and attempts < self.MAX_ATTEMPTS:
                time.sleep(0.05)
                attempts += 1
                result = sock.connect_ex(("0.0.0.0", 8000))
        return attempts, result

    def test_web_server(self):
        with tempfile.TemporaryDirectory(prefix="sourced.ml-test-") as tmpdir:
            os.chdir(tmpdir)
            testfile = "test.txt"
            with open(testfile, "w") as fout:
                fout.write("The Zen of Python, by Tim Peters")
            server = CORSWebServer()
            server.start()

            try:
                attempts, result = self.wait_for_web_server()
                self.assertTrue(attempts < self.MAX_ATTEMPTS or result == 0)
                self.assertEqual(requests.get("http://0.0.0.0:8000/test.txt").text,
                                 "The Zen of Python, by Tim Peters")
            finally:
                server.stop()

    def test_wait(self):
        web_server.start()
        try:
            attempts, result = self.wait_for_web_server()
            self.assertTrue(attempts < self.MAX_ATTEMPTS or result == 0)
            self.assertTrue(web_server.running)
        except:  # nopep8
            web_server.stop()
            raise
        os.environ["PROJECTOR_SERVER_TIME"] = "0"
        wait()
        self.assertFalse(web_server.running)
        web_server.start()
        try:
            attempts, result = self.wait_for_web_server()
            self.assertTrue(attempts < self.MAX_ATTEMPTS or result == 0)
            self.assertTrue(web_server.running)
        finally:
            web_server.stop()

    def test_present_embeddings(self):
        with tempfile.TemporaryDirectory(prefix="sourced.ml-test-") as tmpdir:
            tmpdir = os.path.join(tmpdir, "1", "2")
            present_embeddings(tmpdir, False, ["one", "two"],
                               [(str(i), "x") for i in range(5)],
                               [(i, i) for i in range(5)])
            with open(os.path.join(tmpdir, "id2vec.json")) as fin:
                json.load(fin)
            with open(os.path.join(tmpdir, "id2vec_meta.tsv")) as fin:
                self.assertEqual(fin.read(), "one\ttwo\n0\tx\n1\tx\n2\tx\n3\tx\n4\tx\n")
            with open(os.path.join(tmpdir, "id2vec_data.tsv")) as fin:
                self.assertEqual(fin.read(), "0\t0\n1\t1\n2\t2\n3\t3\n4\t4\n")

    def test_present_embeddings_run_server(self):
        def sweded_which(prog):
            return None

        which = shutil.which
        shutil.which = sweded_which
        browser = os.getenv("BROWSER", "")
        os.environ["BROWSER"] = ""

        try:
            with tempfile.TemporaryDirectory(prefix="sourced.ml-test-") as tmpdir:
                with captured_output() as (stdout, _, _):
                    present_embeddings(tmpdir, True, ["one"],
                                       [str(i) for i in range(5)],
                                       [(i, i) for i in range(5)])
                    with open(os.path.join(tmpdir, "id2vec.json")) as fin:
                        json.load(fin)
                    with open(os.path.join(tmpdir, "id2vec_meta.tsv")) as fin:
                        self.assertEqual(fin.read(), "0\n1\n2\n3\n4\n")
                    with open(os.path.join(tmpdir, "id2vec_data.tsv")) as fin:
                        self.assertEqual(fin.read(), "0\t0\n1\t1\n2\t2\n3\t3\n4\t4\n")
                self.assertIn(
                    "\thttp://projector.tensorflow.org/?config=http://0.0.0.0:8000/id2vec.json\n",
                    stdout.getvalue())
        finally:
            shutil.which = which
            os.environ["BROWSER"] = browser
            web_server.stop()

    def test_stop(self):
        web_server.stop()  # dummy test to avoid partially covered lin in CI
        assert not web_server.running
        web_server.start()
        web_server.stop()
        assert not web_server.running


if __name__ == "__main__":
    unittest.main()
