import os
import unittest
import itertools

from bblfsh import BblfshClient

from collections import Counter
from sourced.ml.transformers import CooccConstructor
from sourced.ml.algorithms import TokenParser
from sourced.ml.algorithms.uast_ids_to_bag import FakeVocabulary
from sourced.ml.tests.models import SOURCE_PY
from sourced.ml.utils import bblfsh_roles


class CooccConstructorTests(unittest.TestCase):
    def setUp(self):
        self.uast = BblfshClient("0.0.0.0:9432").parse(SOURCE_PY).uast
        self.coocc_constructor = CooccConstructor(FakeVocabulary, TokenParser())

    def _traverse_uast_old(self, uast):
        stack = [uast]
        new_stack = []

        while stack:
            for node in stack:
                children = self.coocc_constructor._flatten_children(node)
                tokens = []
                for ch in children:
                    tokens.extend(self.coocc_constructor.token_parser(ch.token))
                token = node.token.strip()
                if token != "" and \
                        bblfsh_roles.IDENTIFIER in node.roles and \
                        bblfsh_roles.QUALIFIED not in node.roles:
                    tokens.extend(self.coocc_constructor.token_parser(token))
                for pair in itertools.permutations(tokens, 2):
                    yield pair

                new_stack.extend(children)

            stack = new_stack
            new_stack = []

    def _flatten_children_old(self, root):
        ids = []
        stack = list(root.children)
        for node in stack:
            if bblfsh_roles.IDENTIFIER in node.roles and \
                    bblfsh_roles.QUALIFIED not in node.roles:
                ids.append(node)
            else:
                stack.extend(node.children)
        return ids

    def test_traverse_uast(self):
        pairs1 = set(Counter(self._traverse_uast_old(self.uast)))
        pairs2 = set(Counter(self.coocc_constructor._traverse_uast(self.uast)))
        self.assertEqual(pairs1, pairs2)

    def test_flatten_children_uast(self):
        ch1 = list(self.coocc_constructor._flatten_children(self.uast))
        ch2 = list(self._flatten_children_old(self.uast))
        self.assertEqual(ch1, ch2)


if __name__ == "__main__":
    unittest.main()
