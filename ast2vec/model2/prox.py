from typing import Dict, Tuple

from ast2vec.model2.proxbase import ProxBase

MATRIX_TYPES = dict()


def register_mat_type(cls):
    """
    Check some conventions for class declaration and add it to MATRIX_TYPES.

    :param cls: Class for proximity matrix model converter.
    """
    base = "Prox"
    assert issubclass(cls, ProxBase), "Must be a subclass of ProxBase."
    assert cls.__name__.startswith(base), "Make sure to start your class name with %s." % (base, )
    MATRIX_TYPES[cls.__name__[len(base):]] = cls

    return cls


@register_mat_type
class ProxGraRep(ProxBase):
    """
    Calculate proximity matrix defined in GraRep algorithm.
    For additional info refer to http://dl.acm.org/citation.cfm?doid=2806416.2806512
    """
    pass


@register_mat_type
class ProxHOPE(ProxBase):
    """
    Calculate proximity matrix defined in HOPE algorithm.
    For additional info refer to http://dl.acm.org/citation.cfm?doid=2939672.2939751
    """
    pass


@register_mat_type
class ProxSwivel(ProxBase):
    """
    Calculate proximity matrix defined in Swivel algorithm (i.e. cooccurence matrix).
    For additional info refer to http://arxiv.org/abs/1602.02215
    """
    def _adj_to_feat(self, role2ind: Dict[int, int], token2ind: Dict[int, int], mat) -> Tuple:
        """
        Convert adjacency matrix to format suitable for Cooccurrences model.

        :param role2ind: Mapping from roles to indices, starting with 0.
        :param token2ind: Mapping from tokens to indices, starting with 0.
        :param mat: Adjacency matrix ('scipy.sparse.coo_matrix') with rows corresponding to
                    node roles followed by node tokens.
        :return: tuple('tokens', 'matrix'). 'tokens' are generalized tokens (usually roles+tokens).
                 'matrix' rows correspond to 'tokens'.
        """
        roles = sorted(role2ind, key=role2ind.get)
        roles = ["RoleId_%d" % role for role in roles]
        tokens = sorted(token2ind, key=token2ind.get)
        return roles + tokens, mat


def prox_entry(args):
    m2p = MATRIX_TYPES[args.matrix_type](num_processes=args.processes, edges=args.edges,
                                         overwrite_existing=args.overwrite_existing)
    m2p.convert(args.input, args.output, args.filter)
