from typing import Dict, Tuple

from ast2vec.coocc import Cooccurrences
from ast2vec.model2.proxbase import ProxBase

MATRIX_TYPES = dict()


def register_mat_type(cls):
    base = "Prox"
    assert issubclass(cls, ProxBase), "Must be a subclass of Repo2ProxBase."
    assert cls.__name__.startswith(base), "Make sure to start your class name with %s." % (base, )
    MATRIX_TYPES[cls.__name__[len(base):]] = cls

    return cls


@register_mat_type
class ProxGraRep(ProxBase):
    pass


@register_mat_type
class ProxHOPE(ProxBase):
    pass


@register_mat_type
class ProxSwivel(ProxBase):
    def _adj_to_feat(self, role2ind: Dict[int, int], token2ind: Dict[int, int], mat) -> Tuple:
        roles = sorted(role2ind, key=role2ind.get)
        roles = ["RoleId_%d" % role for role in roles]
        tokens = sorted(token2ind, key=token2ind.get)
        return roles + tokens, mat


def prox_entry(args):
    processes = args.processes or multiprocessing.cpu_count()
    m2p = MATRIX_TYPES[args.matrix_type](num_processes=processes, edges=args.edges)
    m2p.convert(args.input, args.output, args.filter)
