from modelforge import generate_meta
from modelforge.model import Model, assemble_sparse_matrix, disassemble_sparse_matrix, write_model
from modelforge.models import register_model

import ast2vec


@register_model
class VocabularyCooccurrences(Model):
    """
    Co-occurrence matrix.
    """
    NAME = "vocabulary_co-occurrences"

    def construct(self, matrix):
        self._matrix = matrix

    def _load_tree(self, tree):
        self.construct(matrix=assemble_sparse_matrix(tree["matrix"]))

    def dump(self):
        return """
Matrix: shape: %s non-zero: %d""" % (
            self.matrix.shape, self.matrix.getnnz())

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
        return self._matrix.shape[0]

    def save(self, output, deps=None):
        if not deps:
            deps = tuple()
        self._meta = generate_meta(self.NAME, ast2vec.__version__, *deps)
        write_model(self._meta,
                    {"matrix": disassemble_sparse_matrix(self.matrix)},
                    output)
