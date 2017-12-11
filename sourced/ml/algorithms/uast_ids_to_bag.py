from collections import defaultdict

from sourced.ml.algorithms.token_parser import TokenParser


class FakeVocabulary:
    def __getitem__(self, item):
        return item


class UastIds2Bag:
    """
    Converts a UAST to a bag-of-identifiers.
    """
    def __init__(self, vocabulary, token_parser=None):
        """
        :param vocabulary: The mapping from tokens to bag keys. If None, no mapping is performed.
        :param token_parser: Specify token parser if you want to use a custome one. \
            :class:'TokenParser' is used if it is not specified.
        """
        self._vocabulary = FakeVocabulary() if vocabulary is None else vocabulary
        self._token_parser = TokenParser() if token_parser is None else token_parser

    @property
    def vocabulary(self):
        return self._vocabulary

    def uast_to_bag(self, uast, roles_filter="//*[@roleIdentifier and not(@roleQualified)]"):
        """
        Converts a UAST to a bag-of-words. The weights are identifier frequencies.
        The identifiers are preprocessed by :class:`TokenParser`.

        :param uast: The UAST root node.
        :param roles_filter: The libuast xpath query to filter identifiers.
        :return:
        """
        import bblfsh
        nodes = bblfsh.filter(uast, roles_filter)
        bag = defaultdict(int)
        for node in nodes:
            for sub in self._token_parser.process_token(node.token):
                try:
                    bag[self._vocabulary[sub]] += 1
                except KeyError:
                    continue
        return bag
