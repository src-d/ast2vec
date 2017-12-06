import logging
import os
from operator import add
from uuid import uuid4
from pyspark.sql import column
from pyspark.sql.functions import size, col

from modelforge.meta import generate_meta
from modelforge.model import merge_strings, write_model
from sourced.ml.engine import create_engine, get_tokens
from sourced.ml.repo2.base import UastExtractor, Transformer, Cacher, UastDeserializer, Engine, HeadFiles
from sourced.ml.repo2.token_map import TokenMapTransformer
from sourced.ml.token_parser import TokenParser

import itertools

from sourced.ml.bblfsh_meta import Roles, Node


class SaveASDFCooccModel(Transformer):
    def __init__(self, output, tokens_list, **kwargs):
        super().__init__(**kwargs)
        self.tokens_list = tokens_list
        self.output = output

    def __call__(self, sparce_matrix):
        matrix_count = sparce_matrix.count()
        rows = sparce_matrix.take(matrix_count)

        mat_row, mat_col, mat_weights = zip(*rows)

        write_model(generate_meta("co-occurrences", (1, 0, 0)),
                    {"tokens": merge_strings(self.tokens_list),
                     "matrix": {
                         "shape": (len(self.tokens_list), len(self.tokens_list)),
                         "format": "coo",
                         "data": (mat_weights, (mat_row, mat_col))}
                     },
                    self.output)


class UASTCooccTransformer(Transformer):
    def __init__(self, token2index, token_parser, prune_size=1, **kwargs):
        super().__init__(**kwargs)
        self.token2index = token2index
        self.token_parser = token_parser

        # TODO(zurk): implement pruning
        self.prune_size = prune_size

    def _flatten_children(self, root):
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
                if (node.token.strip() is not None and node.token.strip() != "" and
                            Roles.IDENTIFIER in node.roles and Roles.QUALIFIED not in node.roles):
                    tokens.extend(self.token_parser(node.token))
                for pair in itertools.permutations(tokens, 2):
                    yield pair

                new_stack.extend(children)

            stack = new_stack
            new_stack = []

    def __call__(self, uasts):
        sparce_matrix = uasts.flatMap(self._process_row)\
            .reduceByKey(add)\
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
    if not args.config:
        args.config = []
    engine = create_engine("repos2cooc-%s" % uuid4(), args.repositories, args)

    pipeline = Engine(engine, explain=args.explain)
    pipeline = pipeline.link(HeadFiles())
    pipeline = pipeline.link(UastExtractor(languages=args.languages))
    if args.persist is not None:
        uasts = pipeline.link(Cacher(args.persist))
    else:
        uasts = pipeline

    token_parser = TokenParser()
    token_mapping_transformer = TokenMapTransformer(token_parser)
    tokens, tokens2index = uasts.link(token_mapping_transformer).execute()

    uasts = uasts.link(UastDeserializer())
    tokens_matrix = uasts.link(UASTCooccTransformer(tokens2index, token_parser))
    save_model = tokens_matrix.link(SaveASDFCooccModel(args.output, tokens))
    save_model.execute()

    # from sourced.ml.repo2 import wmhash
    # extractors = [wmhash.__extractors__[s](
    #     args.min_docfreq, **wmhash.__extractors__[s].get_kwargs_fromcmdline(args))
    #     for s in wmhash.__extractors__]
    # tokens_matrix = uasts.link(wmhash.Repo2DocFreq(extractors)).execute()
    # tokens_matrix.explode()

    #tokens_matrix = uasts.link(UASTCooccTransformer(token_mapping, token_parser))
    #tokens_matrix.execute()

    pass