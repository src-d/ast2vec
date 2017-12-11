import argparse
from typing import Union

from modelforge import Model, split_strings, assemble_sparse_matrix, \
    merge_strings, disassemble_sparse_matrix, register_model

from sourced.ml.models import Id2Vec


class BOWBase(Model):
    """
    Base class for weighted bag of words models.
    """
    NAME = "*bow"

    def construct(self, repos, matrix):
        self._repos = repos
        self._matrix = matrix
        return self

    def _load_tree_kwargs(self, tree):
        return dict(repos=split_strings(tree["repos"]),
                    matrix=assemble_sparse_matrix(tree["matrix"]))

    def _load_tree(self, tree):
        self.construct(**self._load_tree_kwargs(tree))

    def dump(self):
        return """Shape: %s
First 10 repos: %s""" % (
            self._matrix.shape, self._repos[:10])

    @property
    def matrix(self):
        """
        Returns the bags as a sparse matrix. Rows are repositories and cols are weights.
        """
        return self._matrix

    @property
    def repos(self):
        return self._repos

    def __getitem__(self, item):
        """
        Returns repository name, word indices and weights for the given
        repository index.

        :param item: Repository index.
        :return: (name, :class:`numpy.ndarray` with word indices, \
                  :class:`numpy.ndarray` with weights)
        """
        data = self._matrix[item]
        return self._repos[item], data.indices, data.data

    def __iter__(self):
        """
        Returns an iterator over the repository indices.
        """
        return iter(range(len(self)))

    def __len__(self):
        """
        Returns the number of repositories.
        """
        return len(self._repos)

    def repository_index_by_name(self, name):
        """
        Looks up repository index by it's name.
        """
        return self._repos_map[name]

    def become_bow(self, vocabulary):
        """Converts this model to BOW."""
        raise NotImplementedError

    @property
    def _repos(self):
        return self.__repos

    @_repos.setter
    def _repos(self, value):
        self.__repos = value
        self._log.info("Building the repository names mapping...")
        self._repos_map = {r: i for i, r in enumerate(self._repos)}

    def _generate_tree(self):
        return {"repos": merge_strings(self._repos),
                "matrix": disassemble_sparse_matrix(self._matrix)}


@register_model
class BOW(BOWBase):
    """
    Weighted bag of words model. Every word is represented with an index.
    Bag = repository. Word = source code identifier. This model depends on
    :class:`sourced.ml.df.DocumentFrequencies`.
    """

    NAME = "bow"

    def construct(self, repos, matrix, tokens):
        super().construct(repos=repos, matrix=matrix)
        self._tokens = tokens
        return self

    def _load_tree_kwargs(self, tree):
        tree_kwargs = super()._load_tree_kwargs(tree)
        tree_kwargs["tokens"] = split_strings(tree["tokens"])
        return tree_kwargs

    @property
    def tokens(self):
        """
        The list of tokens in the model.
        """
        return self._tokens

    def become_bow(self, vocabulary):
        pass

    def save(self, output, deps=tuple()):
        if not deps:
            raise ValueError("You must specify DocumentFrequencies dependency to save BOW.")
        super().save(output, deps)

    def _generate_tree(self):
        tree = super()._generate_tree()
        tree["tokens"] = merge_strings(self.tokens)
        return tree

    def dump(self):
        txt = super().dump()
        txt += "\nFirst 10 tokens: %s" % self.tokens[:10]
        return txt


@register_model
class NBOW(BOWBase):
    """
    Weighted bag of words model. Every word is represented with a vector.
    Bag = repository. Word = source code identifier. This model depends on
    :class:`sourced.ml.id2vec.Id2Vec` and :class:`sourced.ml.df.DocumentFrequencies`.
    """

    NAME = "nbow"

    def become_bow(self, vocabulary: Union[Id2Vec, list]):
        self.NAME = "bow"
        self.__class__ = BOW
        self._tokens = vocabulary.tokens if isinstance(vocabulary, Id2Vec) else vocabulary

    @staticmethod
    def as_bow(nbow: str, id2vec: str) -> BOW:
        bow = NBOW().load(source=nbow)
        if id2vec:
            id2vec = Id2Vec().load(source=id2vec)
        else:
            id2vec = Id2Vec().load(source=bow.get_dependency("id2vec")["uuid"])
        bow.become_bow(id2vec)
        del id2vec
        return bow

    def save(self, output, deps=tuple()):
        if not deps or len(deps) < 2:
            raise ValueError("You must specify DocumentFrequencies and Id2Vec dependencies "
                             "to save NBOW.")
        super().save(output, deps)


def nbow2bow_entry(args: argparse.Namespace):
    bow = NBOW.as_bow(args.nbow, args.id2vec)
    bow.save(args.output)
