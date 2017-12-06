from operator import add

from sourced.ml.repo2.base import Transformer
from sourced.ml.token_parser import TokenParser
from pyspark.sql import functions


class TokenMapper(Transformer):
    def __init__(self,  token_parser, prune_size=1, **kwargs):
        super().__init__(**kwargs)
        self.token_parser = token_parser
        self.prune_size = prune_size

    def get_tokens(self, uasts):
        """
        Get all tokens from uasts.

        :param uasts: UASTsDataFrame from sourced.engine
        :return: DataFrame with new tokens column
        """
        # TODO (zurk): reimplement uast_ids_to_bag.py using new engine and use it instead of get_tokens # nopep8
        return uasts.query_uast('//*[@roleIdentifier and not(@roleIncomplete)]') \
            .extract_tokens() \
            .where(functions.size(functions.col("tokens")) != 0)

    def __call__(self, uasts):
        """
        Make tokens list and token2index mapping from provided uasts.
        token2index is broadcasted dictionary to use it in workers. Maps token to its index.
        """
        tokens = self.get_tokens(uasts).rdd \
            .flatMap(lambda r: [(t, 1) for token in r.tokens for t in self.token_parser(token)]) \
            .reduceByKey(add) \
            .filter(lambda x: x[1] >= self.prune_size) \
            .map(lambda x: x[0])

        self.tokens_number = tokens.count()
        self.tokens = tokens.take(self.tokens_number)
        self.token2index = uasts.rdd.context.broadcast(
            {token: i for i, token in enumerate(self.tokens)})

        return self.tokens, self.token2index
