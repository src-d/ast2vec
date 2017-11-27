from operator import add

from sourced.ml.engine import get_tokens
from sourced.ml.repo2.base import Transformer
from sourced.ml.token_parser import TokenParser


class TokenMapTransformer(Transformer):
    def __init__(self,  token_parser, prune_size=1, **kwargs):
        super().__init__(**kwargs)
        self.token_parser = token_parser
        self.prune_size = prune_size

    def __call__(self, uasts):
        """
        Make tokens list and token2index mapping from provided uasts.
        token2index is broadcasted dictionary to use it in workers. Maps token to its index.
        """
        tokens = get_tokens(uasts).rdd \
            .flatMap(lambda r: [(t, 1) for token in r.tokens for t in self.token_parser(token)]) \
            .reduceByKey(add) \
            .filter(lambda x: x[1] >= self.prune_size) \
            .map(lambda x: x[0])

        self.tokens_number = tokens.count()
        self.tokens = tokens.take(self.tokens_number)
        self.token2index = uasts.rdd.context.broadcast({token: i for i, token in enumerate(self.tokens)})

        return self.tokens, self.token2index
