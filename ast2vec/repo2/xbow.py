from collections import defaultdict

from ast2vec.bblfsh_roles import SIMPLE_IDENTIFIER
from ast2vec.token_parser import TokenParser
from ast2vec.repo2.base import Repo2Base
from ast2vec.token_parser import TokenParser


class Repo2xBOW(Repo2Base):
    """
    Contains common utilities for bag-of-word models.
    """

    def __init__(self, *args, **kwargs):
        self._vocabulary = kwargs.pop("vocabulary")
        super(Repo2xBOW, self).__init__(*args, **kwargs)
        self._token_parser = TokenParser()

    def _uast_to_bag(self, uast):
        stack = [uast]
        bag = defaultdict(int)
        while stack:
            node = stack.pop(0)
            if SIMPLE_IDENTIFIER in node.roles:
                for sub in self._token_parser.process_token(node.token):
                    try:
                        bag[self._vocabulary[sub]] += 1
                    except KeyError:
                        pass
            stack.extend(node.children)
        return bag
