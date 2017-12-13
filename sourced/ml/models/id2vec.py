from modelforge import register_model, Model, split_strings, merge_strings


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
        return self

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

    def items(self):
        """
        Returns the tuples belonging to token -> index mapping.
        """
        return self._token2index.items()

    def __getitem__(self, item):
        """
        Returns the index of the specified processed source code identifier.
        """
        return self._token2index[item]

    def __len__(self):
        """
        Returns the number of tokens in the model.
        """
        return len(self._tokens)

    def _generate_tree(self):
        return {"embeddings": self.embeddings, "tokens": merge_strings(self.tokens)}
