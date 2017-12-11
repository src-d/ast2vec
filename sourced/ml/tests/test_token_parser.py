import pickle
import unittest

from sourced.ml.algorithms import TokenParser, NoTokenParser


class TokenParserTests(unittest.TestCase):
    def setUp(self):
        self.tp = TokenParser(stem_threshold=4, max_token_length=20)

    def test_process_token(self):
        self.assertEqual(
            list(self.tp.split("set /for. *&PrintAll")),
            ["set", "for", "print", "all"])
        self.assertEqual(
            list(self.tp.split("JumpDown not Here")),
            ["jump", "down", "not", "here"])

    def test_split(self):
        self.assertEqual(list(self.tp.split("set for")), ["set", "for"])
        self.assertEqual(list(self.tp.split("set /for.")), ["set", "for"])
        self.assertEqual(list(self.tp.split("NeverHav")), ["never", "hav"])
        self.assertEqual(list(self.tp.split("PrintAll")), ["print", "all"])
        self.assertEqual(list(self.tp.split("PrintAllExcept")), ["print", "all", "except"])
        self.assertEqual(
            list(self.tp.split("print really long line")),
            ["print", "really", "long"])

    def test_stem(self):
        self.assertEqual(self.tp.stem("lol"), "lol")
        self.assertEqual(self.tp.stem("apple"), "appl")
        self.assertEqual(self.tp.stem("orange"), "orang")
        self.assertEqual(self.tp.stem("embedding"), "embed")
        self.assertEqual(self.tp.stem("Alfred"), "Alfred")
        self.assertEqual(self.tp.stem("Pluto"), "Pluto")

    def test_pickle(self):
        tp = pickle.loads(pickle.dumps(self.tp))
        self.assertEqual(tp.stem("embedding"), "embed")


class NoTokenParserTests(unittest.TestCase):
    def setUp(self):
        self.tp = NoTokenParser()

    def test_process_token(self):
        self.assertEqual(self.tp.process_token("abcdef"), ["abcdef"])
        self.assertEqual(self.tp.process_token("abcd_ef"), ["abcd_ef"])
        self.assertEqual(self.tp.process_token("abcDef"), ["abcDef"])


if __name__ == "__main__":
    unittest.main()
