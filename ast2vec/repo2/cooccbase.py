from collections import defaultdict

import numpy
from scipy.sparse import coo_matrix

from ast2vec.bblfsh_roles import SIMPLE_IDENTIFIER
from ast2vec.token_parser import TokenParser
from ast2vec.repo2.base import Repo2Base
from ast2vec.token_parser import TokenParser


class Repo2CooccBase(Repo2Base):
    """
    Converts UASTs to co-occurrence matrices.
    """
    def __init__(self, *args, **kwargs):
        super(Repo2CooccBase, self).__init__(*args, **kwargs)
        self._token_parser = TokenParser()

    def __init__(self, *args, **kwargs):
        super(Repo2CooccBase, self).__init__(*args, **kwargs)
        self._token_parser = TokenParser()

    def convert_uasts(self, file_uast_generator):
        word2ind = self._get_vocabulary()
        dok_matrix = defaultdict(int)
        for file_uast in file_uast_generator:
            self._traverse_uast(file_uast.response.uast, word2ind, dok_matrix)

        n_tokens = len(word2ind)
        mat = coo_matrix((n_tokens, n_tokens), dtype=numpy.float32)

        if n_tokens == 0:
            return [], mat

        mat.row = row = numpy.empty(len(dok_matrix), dtype=numpy.int32)
        mat.col = col = numpy.empty(len(dok_matrix), dtype=numpy.int32)
        mat.data = data = numpy.empty(len(dok_matrix), dtype=numpy.float32)
        for i, (coord, val) in enumerate(sorted(dok_matrix.items())):
            row[i], col[i] = coord
            data[i] = val

        return self._get_result(word2ind, mat)

    def _get_vocabulary(self):
        raise NotImplementedError

    def _get_result(self, word2ind, mat):
        raise NotImplementedError

    def _update_dict(self, generator, word2ind, tokens):
        raise NotImplementedError

    def _flatten_children(self, root):
        ids = []
        stack = list(root.children)
        for node in stack:
            if SIMPLE_IDENTIFIER in node.roles:
                ids.append(node)
            else:
                stack.extend(node.children)
        return ids

    @staticmethod
    def _all2all(words, word2ind):
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                try:
                    wi = word2ind[words[i]]
                    wj = word2ind[words[j]]
                except KeyError:
                    continue
                yield wi, wj, 1
                yield wj, wi, 1

    def _process_node(self, root, word2ind, mat):
        children = self._flatten_children(root)

        tokens = []
        for ch in children:
            self._update_dict(self._token_parser.process_token(ch.token), word2ind, tokens)

        if (root.token.strip() is not None and root.token.strip() != "" and
                SIMPLE_IDENTIFIER in root.roles):
            self._update_dict(self._token_parser.process_token(root.token), word2ind, tokens)

        for triplet in self._all2all(tokens, word2ind):
            mat[(triplet[0], triplet[1])] += triplet[2]
        return children

    def _extract_ids(self, root):
        queue = [root]
        while queue:
            node = queue.pop()
            if SIMPLE_IDENTIFIER in node.roles:
                yield node.token
            for child in node.children:
                queue.append(child)

    def _traverse_uast(self, root, word2ind, dok_mat):
        """
        Traverses UAST and extract the co-occurrence matrix.
        """
        stack = [root]
        new_stack = []

        while stack:
            for node in stack:
                children = self._process_node(node, word2ind, dok_mat)
                new_stack.extend(children)
            stack = new_stack
            new_stack = []
