from modelforge import generate_meta
from modelforge.model import Model, split_strings, assemble_sparse_matrix, write_model, \
    merge_strings, disassemble_sparse_matrix
from modelforge.models import register_model

import ast2vec


@register_model
class Cooccurrences(Model):
    """
    Co-occurrence matrix.
    """
    NAME = "co-occurrences"

    def construct(self, tokens, matrix):
        self._tokens = tokens
        self._matrix = matrix

    def _load_tree(self, tree):
        self.construct(tokens=split_strings(tree["tokens"]),
                       matrix=assemble_sparse_matrix(tree["matrix"]))

    def dump(self):
        return """Number of words: %d
First 10 words: %s
Matrix: shape: %s non-zero: %d""" % (
            len(self.tokens), self.tokens[:10], self.matrix.shape, self.matrix.getnnz())

    @property
    def tokens(self):
        """
        Returns the tokens in the order which corresponds to the matrix's rows and cols.
        """
        return self._tokens

    @property
    def matrix(self):
        """
        Returns the sparse co-occurrence matrix.
        """
        return self._matrix

    def __len__(self):
        """
        Returns the number of tokens in the model.
        """
        return len(self._tokens)

    def save(self, output, deps=None):
        if not deps:
            deps = tuple()
        self._meta = generate_meta(self.NAME, ast2vec.__version__, *deps)
        write_model(self._meta,
                    {"tokens": merge_strings(self.tokens),
                     "matrix": disassemble_sparse_matrix(self.matrix)},
                    output)
