from typing import Iterable, Dict

import numpy

from modelforge import register_model, merge_strings
from sourced.ml.models import DocumentFrequencies


@register_model
class OrderedDocumentFrequencies(DocumentFrequencies):
    """
    Compatible with the original DocumentFrequencies.
    """
    NAME = "ordered_docfreq"

    def construct(self, docs: int, dicts: Iterable[Dict[str, int]]):
        df = {}
        for d in dicts:
            df.update(d)
        super().construct(docs, df)
        self._log.info("Ordering the keys...")
        keys = list(self._df)
        keys.sort()
        self._order = {k: i for i, k in enumerate(keys)}
        return self

    @property
    def order(self):
        return self._order

    def _load_tree(self, tree):
        tokens = None
        original_construct = self.construct
        super_construct = super().construct

        def hacked_construct(docs, tokfreq, **kwargs):
            super_construct(docs=docs, tokfreq=tokfreq)
            nonlocal tokens
            tokens = kwargs["tokens"]

        self.construct = hacked_construct
        try:
            super()._load_tree(tree)
        finally:
            self.construct = original_construct
        self._log.info("Mapping the keys order...")
        self._order = {k: i for i, k in enumerate(tokens)}

    def _generate_tree(self):
        tokens = [None] * len(self)
        freqs = numpy.zeros(len(self), dtype=numpy.float32)
        for k, i in self._order.items():
            tokens[i] = k
            freqs[i] = self._df[k]
        return {"docs": self.docs, "tokens": merge_strings(tokens), "freqs": freqs}