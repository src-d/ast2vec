from itertools import islice
from typing import Iterable, Union, Dict, List

import numpy

from modelforge import Model, split_strings, merge_strings, register_model


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
            for d in tokfreqs:
                df.update(d)
        self._docs = docs
        self._df = df
        return self

    def _load_tree(self, tree: dict, tokens=None):
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
            len(self._df), dict(islice(self._df.items(), 10)), self.docs)

    @property
    def docs(self):
        """
        Returns the number of documents.
        """
        return self._docs

    def prune(self, threshold: int) -> "DocumentFrequencies":
        """
        Removes tokens which occur less than `threshold` times.
        The operation happens *not* in-place - a new model is returned.
        :param threshold: Minimum number of occurrences.
        :return: The new model if the current one had to be changed, otherwise self.
        """
        if threshold < 1:
            raise ValueError("Invalid threshold: %d" % threshold)
        if threshold == 1:
            return self
        self._log.info("Pruning to min %d occurrences", threshold)
        pruned = type(self)()
        pruned._docs = self.docs
        pruned._df = {k: v for k, v in self._df.items() if v >= threshold}
        self._log.info("Size: %d -> %d", len(self), len(pruned))
        pruned._meta = self.meta
        return pruned

    def greatest(self, max_size: int) -> "DocumentFrequencies":
        """
        Truncates the model to most frequent `max_size` tokens.
        The operation happens *not* in-place - a new model is returned.
        :param max_size: The maximum vocabulary size.
        :return: The new model if the current one had to be changed, otherwise self.
        """
        if max_size < 1:
            raise ValueError("Invalid max_size: %d" % max_size)
        if len(self) <= max_size:
            return self
        self._log.info("Pruning to max %d size", max_size)
        pruned = type(self)()
        pruned._docs = self.docs
        freqs = numpy.fromiter(self._df.values(), numpy.int32, len(self))
        chosen = numpy.argpartition(freqs, len(freqs) - max_size)[len(freqs) - max_size:]
        # we need to be deterministic at the cutoff frequency
        # argpartition returns random samples every time
        # so we treat words with the cutoff frequency separately
        chosen.sort()
        border_freq = freqs[chosen].min()
        df = {}
        border_keys = []
        chosen_pos = 0
        chosen_val = chosen[0]
        for i, key in enumerate(self._df):
            if i != chosen_val:
                continue
            if chosen_val > border_freq:
                df[key] = chosen_val
            else:
                border_keys.append(key)
            chosen_pos += 1
            if chosen_pos >= len(chosen):
                break
            chosen_val = chosen[chosen_pos]
        border_keys.sort()
        for k in border_keys[:max_size - len(df)]:
            df[k] = border_freq
        pruned._df = df
        self._log.info("Size: %d -> %d", len(self), len(pruned))
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

    def get(self, item, default=None) -> Union[int, None]:
        """
        Return the document frequency for a given token.

        :param item: The token to query.
        :param default: Returned value in case the token is missing.
        :return: int or `default`
        """
        return self._df.get(item, default)

    def tokens(self) -> List[str]:
        """
        Returns the list of tokens.
        """
        return list(self._df)
