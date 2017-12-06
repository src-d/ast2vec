import itertools
import logging
import operator
from scipy import sparse
from uuid import uuid4

from modelforge.meta import generate_meta
from modelforge.model import merge_strings, write_model
from sourced.ml.coocc import Cooccurrences
from sourced.ml.engine import create_engine
from sourced.ml.repo2.base import UastExtractor, Transformer, Cacher, UastDeserializer, \
    Engine, HeadFiles
from sourced.ml.repo2.token_mapper import TokenMapper
from sourced.ml.token_parser import TokenParser
from sourced.ml.bblfsh_meta import Roles


class CooccModelSaver(Transformer):
    def __init__(self, output, tokens_list, **kwargs):
        super().__init__(**kwargs)
        self.tokens_list = tokens_list
        self.output = output

    def __call__(self, sparce_matrix):
        matrix_count = sparce_matrix.count()
        rows = sparce_matrix.take(matrix_count)

        mat_row, mat_col, mat_weights = zip(*rows)
        tokens_num = len(self.tokens_list)
        matrix = sparse.coo_matrix((mat_weights, (mat_row, mat_col)),
                                   shape=(tokens_num, tokens_num))

        Cooccurrences().construct(self.tokens_list, matrix).save(self.output)


class CooccConstructor(Transformer):
    def __init__(self, token2index, token_parser, prune_size=1, **kwargs):
        super().__init__(**kwargs)
        self.token2index = token2index
        self.token_parser = token_parser

        # TODO(zurk): implement pruning
        self.prune_size = prune_size

    def _flatten_children(self, root):
        # TODO(zurk): refactor using `uast_ids_to_bag.py`
        ids = []
        stack = list(root.children)
        for node in stack:
            if Roles.IDENTIFIER in node.roles and Roles.QUALIFIED not in node.roles:
                ids.append(node)
            else:
                stack.extend(node.children)
        return ids

    def _traverse_uast(self, uast):
        """
        Traverses UAST.
        """

        stack = [uast]
        new_stack = []

        while stack:
            for node in stack:
                children = self._flatten_children(node)
                tokens = []
                for ch in children:
                    tokens.extend(self.token_parser(ch.token))
                token = node.token.strip()
                if node.token.strip() != "" and \
                        Roles.IDENTIFIER in node.roles and \
                        Roles.QUALIFIED not in node.roles:
                    tokens.extend(self.token_parser(node.token))
                for pair in itertools.permutations(tokens, 2):
                    yield pair

                new_stack.extend(children)

            stack = new_stack
            new_stack = []

    def __call__(self, uasts):
        sparce_matrix = uasts.flatMap(self._process_row)\
            .reduceByKey(operator.add)\
            .map(lambda row: (row[0][0], row[0][1], row[1]))
        return sparce_matrix

    def _process_row(self, row):
        for token1, token2 in self._traverse_uast(row.uast):
            try:
                yield (self.token2index.value[token1], self.token2index.value[token2]), 1
            except KeyError:
                # Do not have token1 or token2 in the token2index map
                pass


def repos2coocc_entry(args):
    log = logging.getLogger("repos2cooc")
    engine = create_engine("repos2cooc-%s" % uuid4(), **args.__dict__)

    pipeline = Engine(engine, explain=args.explain)
    pipeline = pipeline.link(HeadFiles())
    pipeline = pipeline.link(UastExtractor(languages=args.languages))
    pipeline = Cacher.maybe(pipeline, args.persist)

    token_parser = TokenParser()
    tokens, tokens2index = pipeline.link(TokenMapper(token_parser)).execute()

    uasts = pipeline.link(UastDeserializer())
    tokens_matrix = uasts.link(CooccConstructor(tokens2index, token_parser))
    save_model = tokens_matrix.link(CooccModelSaver(args.output, tokens))
    save_model.execute()
