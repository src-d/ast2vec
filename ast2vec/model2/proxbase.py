from collections import defaultdict, deque
from itertools import permutations, product
from typing import Dict, Tuple

import numpy
from scipy.sparse import coo_matrix, diags

from ast2vec.coocc import Cooccurrences
from ast2vec.token_parser import TokenParser
from ast2vec.uast import UASTModel
from ast2vec.model2.base import Model2Base

EDGE_TYPES = ["r", "t", "rt", "R", "T", "RT"]
"""
Suppose we have two connected nodes A and B:
r - connect node roles with each other.
t - connect node tokens with each other.
rt - connect node tokens with node roles.
R - connect node A roles with node B roles.
T - connect node A tokens with node B tokens.
RT - connect node A roles(tokens) with node B tokens(roles).
"""


class ProxBase(Model2Base):
    """
    Contains common utilities for proximity matrix models.

    Proximity matrix captures structural information of the graph. Consider A to be adjacency
    matrix, then useful proximity matrices could be A^2, A(A^k-I)/(A-I), etc.
    To get node (entities corresponding to proximity matrix rows) embeddings we just decompose it.
    """
    MODEL_FROM_CLASS = UASTModel
    MODEL_TO_CLASS = Cooccurrences

    def __init__(self, edges=EDGE_TYPES, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.edges = set(edges)
        self._token_parser = TokenParser()
        self._clear()

    def convert_model(self, model) -> Cooccurrences:
        """
        Update attributes by processing UASTs in the input model.
        Then convert it into Cooccurrences model.

        :param model: UASTModel instance.
        :return: Cooccurences model for all UASTs in `model`.
        """
        for uast in model.uasts:
            self._traverse_uast(uast)

        roles_to_roles = defaultdict(int)
        tokens_to_tokens = defaultdict(int)
        roles_to_tokens = defaultdict(int)

        def add_permutations(edge_type, node_items_list, item_to_item):
            if edge_type in self.edges:
                for node_items in node_items_list:
                    for node_item_a, node_item_b in permutations(node_items, 2):
                        item_to_item[(node_item_a, node_item_b)] += 1

        def add_product(edge_type, items_a, items_b, item_to_item):
            if edge_type in self.edges:
                for item_a, item_b in product(items_a, items_b):
                    item_to_item[(item_a, item_b)] += 1

        add_permutations("r", self.roles, roles_to_roles)
        add_permutations("t", self.tokens, tokens_to_tokens)

        for node_roles, node_tokens in zip(self.roles, self.tokens):
            add_product("rt", node_roles, node_tokens, roles_to_tokens)

        for node_a, node_b in self.dok_matrix:
            roles_a = self.roles[node_a]
            roles_b = self.roles[node_b]
            tokens_a = self.tokens[node_a]
            tokens_b = self.tokens[node_b]

            add_product("R", roles_a, roles_b, roles_to_roles)
            add_product("T", tokens_a, tokens_b, tokens_to_tokens)
            add_product("RT", roles_a, tokens_b, roles_to_tokens)

        if roles_to_roles or roles_to_tokens:
            n_roles = len(self.role2ind)
        else:
            n_roles = 0

        if tokens_to_tokens or roles_to_tokens:
            n_tokens = len(self.token2ind)
        else:
            n_tokens = 0

        n_nodes = n_roles + n_tokens
        n_values = len(roles_to_roles) + len(tokens_to_tokens) + len(roles_to_tokens)
        mat = coo_matrix((n_nodes, n_nodes), dtype=numpy.float32)

        mat.row = row = numpy.empty(n_values, dtype=numpy.int32)
        mat.col = col = numpy.empty(n_values, dtype=numpy.int32)
        mat.data = data = numpy.empty(n_values, dtype=numpy.float32)

        def fill_mat(item_to_item, offset):
            for i, (coord, val) in enumerate(sorted(item_to_item.items())):
                row[i + fill_mat.count] = coord[0] + offset[0]
                col[i + fill_mat.count] = coord[1] + offset[1]
                data[i + fill_mat.count] = val
            fill_mat.count += len(item_to_item)
        fill_mat.count = 0

        fill_mat(roles_to_roles, (0, 0))
        fill_mat(roles_to_tokens, (0, n_roles))
        fill_mat(tokens_to_tokens, (n_roles, n_roles))

        mat = coo_matrix(mat + mat.T - diags(mat.diagonal()))
        tokens, mat = self._adj_to_feat(self.role2ind, self.token2ind, mat)
        self._clear()

        prox = Cooccurrences()
        prox.construct(tokens=tokens, matrix=mat)
        return prox

    def _adj_to_feat(self, role2ind: Dict[int, int], token2ind: Dict[int, int], mat) -> Tuple:
        """
        This must be implemented in the child classes.

        :param role2ind: Mapping from roles to indices, starting with 0.
        :param token2ind: Mapping from tokens to indices, starting with 0.
        :param mat: Adjacency matrix ('scipy.sparse.coo_matrix') with rows corresponding to
                    node roles followed by node tokens.
        :return: tuple('tokens', 'matrix'). 'tokens' are generalized tokens (usually roles+tokens).
                 'matrix' rows correspond to 'tokens'.
        """
        raise NotImplementedError

    def _clear(self):
        """
        Release memory.
        """
        self.roles = list()
        self.tokens = list()
        self.role2ind = dict()
        self.token2ind = dict()
        self.dok_matrix = defaultdict(int)

    def _traverse_uast(self, root) -> None:
        """
        Traverse UAST and extract adjacency matrix.

        :param root: UAST root node.
        :return: None
        """
        n_nodes = len(self.roles)
        queue = deque([(root, n_nodes)])  # (node, node_idx)

        while queue:
            node, node_idx = queue.popleft()
            node_tokens = list(self._token_parser.process_token(node.token))

            for role in node.roles:
                self.role2ind.setdefault(role, len(self.role2ind))
            for token in node_tokens:
                self.token2ind.setdefault(token, len(self.token2ind))

            self.roles.append([self.role2ind[role] for role in node.roles])
            self.tokens.append([self.token2ind[token] for token in node_tokens])

            for ch in node.children:
                n_nodes += 1
                self.dok_matrix[(node_idx, n_nodes)] += 1
                queue.append((ch, n_nodes))
