from ast2vec.coocc import Cooccurrences
from ast2vec.repo2.base import repos2_entry, repo2_entry
from ast2vec.repo2.coocc import Repo2CooccTransformer
from ast2vec.repo2.proxbase import Repo2ProxBase

MATRIX_TYPES = dict()


def register_mat_type(cls):
    base = "Repo2Prox"
    assert issubclass(cls, Repo2ProxBase), "Must be a subclass of Repo2ProxBase."
    assert cls.__name__.startswith(base), "Make sure to start your class name with %s." % (base, )
    MATRIX_TYPES[cls.__name__[len(base):]] = cls

    cls.MODEL_CLASS = Cooccurrences

    return cls


@register_mat_type
class Repo2ProxGraRep(Repo2ProxBase):
    pass


@register_mat_type
class Repo2ProxHOPE(Repo2ProxBase):
    pass


@register_mat_type
class Repo2ProxSwivel(Repo2ProxBase):
    def _adj_to_feat(self, role2ind, token2ind, mat):
        roles = sorted(role2ind, key=role2ind.get)
        roles = ["RoleId_%d" % role for role in roles]
        tokens = sorted(token2ind, key=token2ind.get)
        return roles + tokens, mat


def repo2prox_entry(args):
    Repo2CooccTransformer.WORKER_CLASS = MATRIX_TYPES[args.matrix_type]
    return repo2_entry(args, Repo2CooccTransformer, "matrix_type")


def repos2prox_entry(args):
    Repo2CooccTransformer.WORKER_CLASS = MATRIX_TYPES[args.matrix_type]
    return repos2_entry(args, Repo2CooccTransformer, "matrix_type")
