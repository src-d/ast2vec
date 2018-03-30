from typing import Iterable, Tuple

import bblfsh

from sourced.ml.algorithms.uast_ids_to_bag import UastIds2Bag


class MergeRoles:
    def __getitem__(self, roles):
        return " ".join(bblfsh.role_name(r) for r in sorted(roles))


class Uast2RoleIdPairs(UastIds2Bag):
    """
    Converts a UAST to a list of pairs. Pair is identifier and role, where role is Node role
    where identifier was found.

    Be careful, __call__ is overridden here and return list instead of bag-of-words (dist).
    """

    def __init__(self, token2index=None, token_parser=None, roles2index=None):
        """
        :param token2index: The mapping from tokens to token key. If None, no mapping is performed.
        :param token_parser: Specify token parser if you want to use a custom one. \
            :class:'TokenParser' is used if it is not specified.
        :param roles2index: The mapping from role index list to keys.
        """
        super().__init__(token2index=token2index, token_parser=token_parser)
        self._roles2index = MergeRoles() if roles2index is None else roles2index

    @property
    def roles2index(self):
        return self._roles2index

    def __call__(self, uast: bblfsh.Node) -> Iterable[Tuple[str, str]]:
        """
        Converts a UAST to a list of identifier, role pairs.
        The tokens are preprocessed by _token_parser.

        :param uast: The UAST root node.
        :return: a list of identifier, role pairs.
        """
        yield from self._process_uast(uast, [])

    def _process_uast(self, node, ancestors):
        identifier_id = bblfsh.role_id("IDENTIFIER")
        operator_id = bblfsh.role_id("OPERATOR")
        exclude = {
            bblfsh.role_id("EXPRESSION"),
            identifier_id,
            bblfsh.role_id("LEFT"),
            bblfsh.role_id("QUALIFIED"),
            bblfsh.role_id("BINARY"),
            bblfsh.role_id("ASSIGNMENT"),
        }

        if identifier_id in node.roles and node.token:
            roles = set(node.roles)
            indx = -1
            # We skip all Nodes with EXPRESSION role or EXPRESSION, IDENTIFIER or
            # EXPRESSION, IDENTIFIER roles.
            # For them we take first parent Node from stack with another Role set.
            while not (roles - exclude and operator_id not in roles):
                roles = set(ancestors[indx].roles)
                indx -= 1
            for sub in self._token_parser.process_token(node.token):
                try:
                    yield (self._token2index[sub], self._roles2index[roles])
                except KeyError:
                    continue

        ancestors.append(node)
        for child in node.children:
            yield from self._process_uast(child, ancestors)
        ancestors.pop()
