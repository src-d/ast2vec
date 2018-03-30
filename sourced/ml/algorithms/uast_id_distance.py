from collections import namedtuple
from typing import Iterable, Tuple, Union
from itertools import combinations

import bblfsh

from sourced.ml.algorithms.uast_ids_to_bag import UastIds2Bag


Point = namedtuple("Point", ["token", "info"])


class Uast2IdDistance(UastIds2Bag):
    """
    Converts a UAST to a list of identifiers pair and UAST distance between.
    distance should be defined in inheritors class

    Be careful, __call__ is overridden here and return list instead of bag-of-words (dist).
    """

    DEFAULT_MAX_DISTANCE = 10  # to avoid collecting all distances we skip too big ones
    IDENTIFIER = bblfsh.role_id("IDENTIFIER")

    def __init__(self, token2index=None, token_parser=None, max_distance=DEFAULT_MAX_DISTANCE):
        """
        :param token2index: The mapping from tokens to token key. If None, no mapping is performed.
        :param token_parser: Specify token parser if you want to use a custom one. \
            :class:'TokenParser' is used if it is not specified.
        """
        super().__init__(token2index=token2index, token_parser=token_parser)
        self.max_distance = max_distance

    def __call__(self, uast: bblfsh.Node) -> Iterable[Tuple[str, str, int]]:
        """
        Converts a UAST to a list of identifiers pair and UAST distance between.
        The tokens are preprocessed by _token_parser.

        :param uast: The UAST root node.
        :return: a list of (from identifier, to identifier) and distance pairs.
        """
        for point1, point2 in combinations(self.process_uast(uast), 2):
            if point1.token == point2.token:
                continue  # We do not want to calculate distance between the same identifiers
            distance = self.distance(point1, point2)
            if distance < self.max_distance:
                yield ((point1.token, point2.token), distance)

    def distance(self, point1: Point, point2: Point) -> Union[int, float]:
        """
        Calculate distance between two points. A point can be anything. self._process_uast returns
        list of points in the specific class.

        :return: Distance between two points.
        """
        raise NotImplemented

    def process_uast(self, node: bblfsh.Node) -> Iterable[Point]:
        """
        Converts uast to points list. A point can be anything you need to calculate distance.
        """
        raise NotImplemented


class Uast2IdTreeDistance(Uast2IdDistance):
    """
    Converts a UAST to a list of identifiers pair and UAST tree distance between.

    Be careful, __call__ is overridden here and return list instead of bag-of-words (dist).
    """
    def process_uast(self, node: bblfsh.Node, ancestors=None) -> Iterable[Point]:
        if ancestors is None:
            ancestors = []
        if self.IDENTIFIER in node.roles and node.token:
            for sub in self._token_parser.process_token(node.token):
                try:
                    yield Point(self._token2index[sub], list(ancestors))
                except KeyError:
                    continue

        ancestors.append(node)
        for child in node.children:
            yield from self.process_uast(child, ancestors)
        ancestors.pop()

    def distance(self, point1: Point, point2: Point) -> int:
        i = 0
        ancestors1 = point1.info
        ancestors2 = point2.info
        for i, (ancestor1, ancestor2) in enumerate(zip(ancestors1, ancestors2)):
            if ancestor1 != ancestor2:
                break
        distance = self.calc_tree_distance(i, len(ancestors1), len(ancestors2))
        return distance

    @staticmethod
    def calc_tree_distance(last_common_level, level1, level2):
        return level1 + level2 - 2 * last_common_level


class Uast2IdLineDistance(Uast2IdDistance):
    """
    Converts a UAST to a list of identifiers pair and code line distance between where applicable.

    Be careful, __call__ is overridden here and return list instead of bag-of-words (dist).
    """

    def process_uast(self, node, last_position=0):
        if node.start_position.line != 0:
            # A lot of Nodes do not have position
            # It is good heuristic to take the last Node in tree with a position.
            last_position = node.start_position.line

        if self.IDENTIFIER in node.roles and node.token:
            for sub in self._token_parser.process_token(node.token):
                try:
                    yield Point(self._token2index[sub], last_position)
                except KeyError:
                    continue

        for child in node.children:
            yield from self.process_uast(child, last_position)

    def distance(self, point1: Point, point2: Point):
        return abs(point1.info - point2.info)  # info is equal to line number in this case
