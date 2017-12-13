from typing import Union

from modelforge import register_model, Model, split_strings, assemble_sparse_matrix, \
    merge_strings, disassemble_sparse_matrix


@register_model
class Topics(Model):
    NAME = "topics"

    @property
    def tokens(self):
        return self._tokens

    @property
    def topics(self):
        """
        May be None if no topics are labeled.
        """
        return self._topics

    @property
    def matrix(self):
        """
        Rows: tokens
        Columns: topics
        """
        return self._matrix

    def construct(self, tokens: list, topics: Union[list, None], matrix):
        if len(tokens) != matrix.shape[1]:
            raise ValueError("Tokens and matrix do not match.")
        self._tokens = tokens
        self._topics = topics
        self._matrix = matrix
        return self

    def _load_tree(self, tree: dict) -> None:
        self.construct(split_strings(tree["tokens"]),
                       split_strings(tree["topics"]) if tree["topics"] else None,
                       assemble_sparse_matrix(tree["matrix"]))

    def dump(self) -> str:
        res = "%d topics, %d tokens\nFirst 10 tokens: %s\nTopics: " % (
            self.matrix.shape + (self.tokens[:10],))
        if self.topics is not None:
            res += "labeled, first 10: %s\n" % self.topics[:10]
        else:
            res += "unlabeled\n"
        nnz = self.matrix.getnnz()
        res += "non-zero elements: %d  (%f)" % (
            nnz, nnz / (self.matrix.shape[0] * self.matrix.shape[1]))
        return res

    def _generate_tree(self):
        return {"tokens": merge_strings(self.tokens),
                "topics": merge_strings(self.topics) if self.topics is not None else False,
                "matrix": disassemble_sparse_matrix(self.matrix)}

    def __len__(self):
        """
        Returns the number of topics.
        """
        return self.matrix.shape[0]

    def __getitem__(self, item):
        """
        Returns the keywords sorted by significance from topic index.
        """
        row = self.matrix[item]
        nnz = row.nonzero()[1]
        pairs = [(-row[0, i], i) for i in nnz]
        pairs.sort()
        return [(self.tokens[pair[1]], -pair[0]) for pair in pairs]

    def label_topics(self, labels):
        if len(labels) != len(self):
            raise ValueError("Sizes do not match: %d != %d" % (len(labels), len(self)))
        if not isinstance(labels[0], str):
            raise TypeError("Labels must be strings")
        self._topics = list(labels)
