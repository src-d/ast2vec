import logging
import math
from collections import defaultdict

from scipy.sparse import csr_matrix

from ast2vec.bow import NBOW
from ast2vec.df import DocumentFrequencies
from ast2vec.id2vec import Id2Vec
from ast2vec.repo2.base import RepoTransformer, repos2_entry, repo2_entry
from ast2vec.repo2.xbow import Repo2xBOW
from modelforge.backends import create_backend


class Repo2nBOW(Repo2xBOW):
    """
    Implements the step repository -> :class:`ast2vec.nbow.NBOW`.
    """
    MODEL_CLASS = NBOW

    def __init__(self, id2vec, docfreq, tempdir=None, linguist=None,
                 log_level=logging.INFO, bblfsh_endpoint=None,
                 timeout=Repo2xBOW.DEFAULT_BBLFSH_TIMEOUT):
        super(Repo2nBOW, self).__init__(
            tempdir=tempdir, linguist=linguist, log_level=log_level,
            bblfsh_endpoint=bblfsh_endpoint, timeout=timeout,
            vocabulary=id2vec)
        self._id2vec = id2vec
        self._docfreq = docfreq

    @property
    def id2vec(self):
        return self._id2vec

    @property
    def docfreq(self):
        return self._docfreq

    def convert_uasts(self, file_uast_generator):
        freqs = defaultdict(int)
        for file_uast in file_uast_generator:
            bag = self._uast_to_bag(file_uast.response.uast)
            for key, freq in bag.items():
                freqs[key] += freq
        missing = []
        vocabulary = self._id2vec.tokens
        for key, val in freqs.items():
            try:
                freqs[key] = math.log(1 + val) * math.log(
                    self._docfreq.docs / self._docfreq[vocabulary[key]])
            except KeyError:
                missing.append(key)
        for key in missing:
            del freqs[key]
        return dict(freqs)


class Repo2nBOWTransformer(RepoTransformer):
    """
    Wrap the step: repository -> :class:`ast2vec.nbow.NBOW`.
    """
    WORKER_CLASS = Repo2nBOW

    def __init__(self, id2vec=None, docfreq=None, gcs_bucket=None, **kwargs):
        if gcs_bucket:
            backend = create_backend("gcs", "bucket=" + gcs_bucket)
        else:
            backend = None
        self._id2vec = kwargs["id2vec"] = Id2Vec().load(id2vec or None, backend=backend)
        self._df = kwargs["docfreq"] = DocumentFrequencies().load(docfreq or None, backend=backend)
        super(Repo2nBOWTransformer, self).__init__(**kwargs)

    def dependencies(self):
        return self._df, self._id2vec

    def result_to_model_kwargs(self, result, url_or_path: str):
        if not result:
            raise ValueError("Empty bag")
        csr_data = []
        csr_indices = []
        csr_indptr = [0, len(result)]
        for key, val in sorted(result.items()):
            csr_data.append(val)
            csr_indices.append(key)
        return {"repos": [url_or_path],
                "matrix": csr_matrix((csr_data, csr_indices, csr_indptr),
                                     shape=(1, len(self._id2vec.tokens)))}


def repo2nbow_entry(args):
    return repo2_entry(args, Repo2nBOWTransformer)


def repos2nbow_entry(args):
    return repos2_entry(args, Repo2nBOWTransformer)
