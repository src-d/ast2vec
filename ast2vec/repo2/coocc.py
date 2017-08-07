from modelforge.meta import generate_meta
from modelforge.model import disassemble_sparse_matrix, merge_strings

import ast2vec
from ast2vec.coocc import Cooccurrences
from ast2vec.repo2.base import RepoTransformer, repos2_entry, repo2_entry
from ast2vec.repo2.cooccbase import Repo2CooccBase


class Repo2Coocc(Repo2CooccBase):
    """
    Convert UAST to tuple (list of unique words, list of triplets (word1_ind,
    word2_ind, cnt)).
    """
    MODEL_CLASS = Cooccurrences

    def _get_vocabulary(self):
        return {}

    def _get_result(self, word2ind, mat):
        words = [p[1] for p in sorted((word2ind[w], w) for w in word2ind)]
        return words, mat

    def _update_dict(self, generator, word2ind, tokens):
        for token in generator:
            word2ind.setdefault(token, len(word2ind))
            tokens.append(token)


class Repo2CooccTransformer(RepoTransformer):
    WORKER_CLASS = Repo2Coocc

    def dependencies(self):
        return []

    def result_to_model_kwargs(self, result, url_or_path):
        vocabulary, matrix = result
        if not vocabulary:
            raise ValueError("Empty vocabulary")
        return {"tokens": vocabulary, "matrix": matrix}


def repo2coocc_entry(args):
    return repo2_entry(args, Repo2CooccTransformer)


def repos2coocc_entry(args):
    return repos2_entry(args, Repo2CooccTransformer)
