from collections import defaultdict

import bblfsh

from sourced.ml.algorithms import TokenParser, NoopTokenParser
from sourced.ml.algorithms.uast_to_bag import Uast2BagBase


class FakeVocabulary:
    def __getitem__(self, item):
        return item


class UastTokens2Bag(Uast2BagBase):
    """
    Converts a UAST to a weighed bag of tokens via xpath.
    """

    XPATH = None  # Should be overridden in child class

    def __init__(self, token2index=None, token_parser=None):
        """
        :param token2index: The mapping from tokens to bag keys. If None, no mapping is performed.
        :param token_parser: Specify token parser if you want to use a custom one. \
            :class:'NoopTokenParser' is used if it is not specified.
        """
        self._token2index = FakeVocabulary() if token2index is None else token2index
        self._token_parser = NoopTokenParser() if token_parser is None else token_parser

    @property
    def token_parser(self):
        return self._token_parser

    @property
    def token2index(self):
        return self._token2index

    def __call__(self, uast):
        """
        Converts a UAST to a weighed bag-of-words. The weights are words frequencies.
        The tokens are preprocessed by _token_parser.

        :param uast: The UAST root node.
        :return:
        """
        nodes = bblfsh.filter(uast, self.XPATH)
        bag = defaultdict(int)
        for node in nodes:
            for sub in self._token_parser.process_token(node.token):
                try:
                    bag[self._token2index[sub]] += 1
                except KeyError:
                    continue
        return bag


class UastIds2Bag(UastTokens2Bag):
    """
    Converts a UAST to a bag-of-identifiers.
    """

    XPATH = "//*[@roleIdentifier and not(@roleQualified)]"

    def __init__(self, token2index=None, token_parser=None):
        """
        :param token2index: The mapping from tokens to bag keys. If None, no mapping is performed.
        :param token_parser: Specify token parser if you want to use a custom one. \
            :class:'TokenParser' is used if it is not specified.
        """
        token_parser = TokenParser() if token_parser is None else token_parser
        super().__init__(token2index, token_parser)
