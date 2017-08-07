from modelforge import generate_meta
from modelforge.model import Model, split_strings, write_model, merge_strings
from modelforge.models import register_model

import ast2vec


@register_model
class Id2Vec(Model):
    """
    id2vec model - source code identifier embeddings.
    """
    NAME = "id2vec"

    def construct(self, embeddings, tokens):
        self._embeddings = embeddings
        self._tokens = tokens
        self._log.info("Building the token index...")
        self._token2index = {w: i for i, w in enumerate(self._tokens)}

    def _load_tree(self, tree):
        self.construct(embeddings=tree["embeddings"].copy(),
                       tokens=split_strings(tree["tokens"]))

    def dump(self):
        return """Shape: %s
First 10 words: %s""" % (
            self.embeddings.shape, self.tokens[:10])

    @property
    def embeddings(self):
        """
        :class:`numpy.ndarray` with the embeddings of shape
        (N tokens x embedding dims).
        """
        return self._embeddings

    @property
    def tokens(self):
        """
        List with the processed source code identifiers.
        """
        return self._tokens

    def __getitem__(self, item):
        """
        Returns the index of the specified processed source code identifier.
        """
        return self._token2index[item]

    def save(self, output, deps=None):
        if not deps:
            deps = tuple()
        self._meta = generate_meta(self.NAME, ast2vec.__version__, *deps)
        write_model(self._meta,
                    {"embeddings": self.embeddings, "tokens": merge_strings(self.tokens)},
                    output)
