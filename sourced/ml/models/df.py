from typing import Iterable, Union, Dict

from modelforge import Model, split_strings, merge_strings, register_model
import numpy


@register_model
class DocumentFrequencies(Model):
    """
    Document frequencies - number of times a source code identifier appeared
    in different repositories. Each repository counts only once.
    """
    NAME = "docfreq"

    def construct(self, docs: int, tokfreqs: Union[Iterable[Dict[str, int]], Dict[str, int]]):
        """
        Initializes this model.
        :param docs: The number of documents.
        :param tokfreqs: The dictionary of token -> frequency or the iterable collection of such
                         dictionaries.
        :return: self
        """
        if isinstance(tokfreqs, dict):
            df = tokfreqs
        else:
            df = {}
            self.quant_map = tokfreqs[-1]
            if self.quant_map:
                for d in tokfreqs[:-1]:
                    df.update(d)
            else:
                try:
                    for d in tokfreqs:
                        df.update(d)
                except TypeError:
                    pass
        self._docs = docs
        self._df = df
        return self

    def _load_tree(self, tree, tokens=None):
        if tokens is None:
            tokens = split_strings(tree["tokens"])
        freqs = tree["freqs"]
        self._log.info("Building the docfreq dictionary...")
        tokfreq = dict(zip(tokens, freqs))
        self.construct(docs=tree["docs"], tokfreqs=tokfreq)

    def _generate_tree(self):
        tokens = self.tokens()
        freqs = numpy.array([self._df[t] for t in tokens], dtype=numpy.float32)
        return {"docs": self.docs, "tokens": merge_strings(tokens), "freqs": freqs}

    def dump(self):
        return """Number of words: %d
First 10 words: %s
Number of documents: %d""" % (
            len(self._df), self.tokens()[:10], self.docs)

    @property
    def docs(self):
        """
        Returns the number of documents.
        """
        return self._docs

    @property
    def docfreq(self):
        """
        Returns document frequency dictionary.
        """
        return self._df

    def prune(self, threshold: int):
        """
        Removes tokens which occur less than `threshold` times.
        The operation happens *not* in-place - a new model is returned.
        :param threshold: Minimum number of occurrences.
        :return: the pruned model.
        """
        self._log.info("Pruning to min %d occurrences", threshold)
        pruned = DocumentFrequencies()
        pruned._docs = self.docs
        pruned._df = {k: v for k, v in self._df.items() if v >= threshold}
        pruned._meta = self.meta
        return pruned

    def __getitem__(self, item):
        return self._df[item]

    def __iter__(self):
        return iter(self._df.items())

    def __len__(self):
        """
        Returns the number of tokens in the model.
        """
        return len(self._df)

    def get(self, item, default):
        """
        Return the document frequency for a given token.

        :param item: The token to query.
        :param default: Returned value in case the token is missing.
        :return: int
        """
        return self._df.get(item, default)

    def tokens(self):
        """
        Returns the sorted list of tokens.
        """
        return sorted(self._df)
