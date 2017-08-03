from collections import defaultdict, deque
from itertools import permutations, product
import logging
from typing import Dict, List, Tuple

import numpy
from scipy.sparse import coo_matrix, diags

from ast2vec.repo2.base import Repo2Base

EDGE_TYPES = ["r", "t", "rt", "R", "T", "RT"]


class Repo2ProxBase(Repo2Base):
    """
    Contains common utilities for proximity matrix models.
    """

    def __init__(self, edges=EDGE_TYPES, tempdir=None, linguist=None, log_level=logging.INFO,
                 bblfsh_endpoint=None, timeout=Repo2Base.DEFAULT_BBLFSH_TIMEOUT):
        super(Repo2ProxBase, self).__init__(
            tempdir=tempdir, linguist=linguist, log_level=log_level,
            bblfsh_endpoint=bblfsh_endpoint, timeout=timeout)
        self._edges = set(edges)

    def convert_uasts(self, file_uast_generator):
        roles = list()
        tokens = list()
        role2ind = dict()
        token2ind = dict()
        dok_matrix = defaultdict(int)

        for file_uast in file_uast_generator:
            self._traverse_uast(file_uast.response.uast, roles, tokens,
                                role2ind, token2ind, dok_matrix)

        mat = self._convert_adj_mat(roles, tokens, role2ind, token2ind, dok_matrix)
        return self._adj_to_feat(role2ind, token2ind, mat)

    def _adj_to_feat(self, role2ind, token2ind, mat):
        raise NotImplementedError

    def _convert_adj_mat(self, roles: List[List[int]], tokens: List[List[int]],
                         role2ind: Dict[int, int], token2ind: Dict[str, int],
                         dok_mat: Dict[Tuple[int, int], int]):
        """
        Convert adjacency matrix from node-to-node to role-to-token. This will result in some
        information loss about graph structure.

        :param roles: List of node roles.
        :param tokens: List of node tokens.
        :param role2ind: Mapping from roles to indices, starting with 0.
        :param token2ind: Mapping from tokens to indices, starting with 0.
        :param dok_mat: Dict storing connected node pairs.
        :return: Adjacency matrix ('scipy.sparse.coo_matrix') with rows corresponding to
                 node roles followed by node tokens.
        """
        roles_to_roles = defaultdict(int)
        tokens_to_tokens = defaultdict(int)
        roles_to_tokens = defaultdict(int)

        def add_permutations(edge_type, node_items_list, item_to_item):
            if edge_type in self._edges:
                for node_items in node_items_list:
                    for node_item_a, node_item_b in permutations(node_items, 2):
                        item_to_item[(node_item_a, node_item_b)] += 1

        def add_product(edge_type, items_a, items_b, item_to_item):
            if edge_type in self._edges:
                for item_a, item_b in product(items_a, items_b):
                    item_to_item[(item_a, item_b)] += 1

        add_permutations("r", roles, roles_to_roles)
        add_permutations("t", tokens, tokens_to_tokens)

        for node_roles, node_tokens in zip(roles, tokens):
            add_product("rt", node_roles, node_tokens, roles_to_tokens)

        for node_a, node_b in dok_mat:
            roles_a = roles[node_a]
            roles_b = roles[node_b]
            tokens_a = tokens[node_a]
            tokens_b = tokens[node_b]

            add_product("R", roles_a, roles_b, roles_to_roles)
            add_product("T", tokens_a, tokens_b, tokens_to_tokens)
            add_product("RT", roles_a, tokens_b, tokens_to_tokens)

        if roles_to_roles or roles_to_tokens:
            n_roles = len(role2ind)
        else:
            n_roles = 0

        if tokens_to_tokens or roles_to_tokens:
            n_tokens = len(token2ind)
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
        return mat

    def _traverse_uast(self, root, roles: List[List[int]], tokens: List[List[int]],
                       role2ind: Dict[int, int], token2ind: Dict[str, int],
                       dok_mat: Dict[Tuple[int, int], int]) -> None:
        """
        Traverse UAST and extract adjacency matrix.

        :param root: UAST root node.
        :param roles: Role indices for traversed nodes will be appended to this list.
        :param tokens: Token indices for traversed nodes will be appended to this list.
        :param role2ind: Mapping from roles to indices, starting with 0.
        :param token2ind: Mapping from tokens to indices, starting with 0.
        :param dok_mat: Dict for storing connected node pairs.
        :return: None
        """
        n_nodes = len(roles)
        queue = deque([(root, n_nodes)])  # (node, node_idx)

        while queue:
            node, node_idx = queue.popleft()
            node_tokens = list(self._process_token(node.token))

            for role in node.roles:
                role2ind.setdefault(role, len(role2ind))
            for token in node_tokens:
                token2ind.setdefault(token, len(token2ind))

            roles.append([role2ind[role] for role in node.roles])
            tokens.append([token2ind[token] for token in node_tokens])

            for ch in node.children:
                n_nodes += 1
                dok_mat[(node_idx, n_nodes)] += 1
                queue.append((ch, n_nodes))
