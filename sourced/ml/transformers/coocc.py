import itertools
import operator

from scipy import sparse
from bblfsh import Node
from pyspark.rdd import PipelinedRDD

from sourced.ml.models import Cooccurrences
from sourced.ml.transformers import Transformer
from sourced.ml.utils import bblfsh_roles


class CooccModelSaver(Transformer):
    def __init__(self, output, tokens_list, **kwargs):
        super().__init__(**kwargs)
        self.tokens_list = tokens_list
        self.output = output

    def __call__(self, sparse_matrix: PipelinedRDD):
        """
        Saves Cooccurrences asdf model to disk.

        :param sparse_matrix: rdd with 3 columns: matrix row, matrix column,  cell value. Use
            :class:`.CooccConstructor` to construct RDD from uasts.
        :return:
        """
        matrix_count = sparse_matrix.count()
        rows = sparse_matrix.take(matrix_count)

        mat_row, mat_col, mat_weights = zip(*rows)
        tokens_num = len(self.tokens_list)
        matrix = sparse.coo_matrix((mat_weights, (mat_row, mat_col)),
                                   shape=(tokens_num, tokens_num))

        Cooccurrences().construct(self.tokens_list, matrix).save(self.output)


class CooccConstructor(Transformer):
    """
    Co-occurrence matrix calculation transformer.
    You can find an algorithm full description in :ref:`coocc.md`
    """
    def __init__(self, token2index, token_parser, **kwargs):
        super().__init__(**kwargs)
        self.token2index = token2index
        self.token_parser = token_parser

    def _flatten_children(self, root):
        ids = []
        stack = list(root.children)
        while stack:
            node = stack.pop(0)
            if bblfsh_roles.IDENTIFIER in node.roles and \
                    bblfsh_roles.QUALIFIED not in node.roles:
                ids.append(node)
            else:
                stack.extend(node.children)
        return ids

    def _traverse_uast(self, uast: Node):
        stack = [uast]

        while stack:
            node = stack.pop(0)
            children = self._flatten_children(node)
            tokens = [token for ch in children for token in self.token_parser(ch.token)]

            if bblfsh_roles.IDENTIFIER in node.roles and \
                    bblfsh_roles.QUALIFIED not in node.roles:
                tokens.extend(self.token_parser(node.token))
            for pair in itertools.permutations(tokens, 2):
                yield pair

            stack.extend(children)

    def __call__(self, uasts):
        sparse_matrix = uasts.flatMap(self._process_row)\
            .reduceByKey(operator.add)\
            .map(lambda row: (row[0][0], row[0][1], row[1]))
        return sparse_matrix

    def _process_row(self, row):
        for token1, token2 in self._traverse_uast(row.uast):
            try:
                yield (self.token2index.value[token1], self.token2index.value[token2]), 1
            except KeyError:
                # Do not have token1 or token2 in the token2index map
                pass
