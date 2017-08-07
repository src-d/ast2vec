from modelforge import generate_meta
from modelforge.model import Model, split_strings, write_model, merge_strings
from modelforge.models import register_model
import numpy

import ast2vec


@register_model
class DocumentFrequencies(Model):
    """
    Document frequencies - number of times a source code identifier appeared
    in different repositories. Each repository counts only once.
    """
    NAME = "docfreq"

    def construct(self, docs, tokens, freqs):
        self._docs = docs
        self._log.info("Building the docfreq dictionary...")
        self._df = dict(zip(tokens, freqs))

    def _load_tree(self, tree):
        self.construct(docs=tree["docs"], tokens=split_strings(tree["tokens"]),
                       freqs=tree["freqs"])

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

    def __getitem__(self, item):
        return self._df[item]

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

    def __len__(self):
        """
        Returns the number of tokens in the model.
        """
        return len(self._df)

    def save(self, output, deps=None):
        if not deps:
            deps = tuple()
        self._meta = generate_meta(self.NAME, ast2vec.__version__, *deps)
        tokens = self.tokens()
        freqs = numpy.array([self._df[t] for t in tokens], dtype=numpy.float32)
        write_model(self._meta,
                    {"docs": self.docs, "tokens": merge_strings(tokens), "freqs": freqs},
                    output)
