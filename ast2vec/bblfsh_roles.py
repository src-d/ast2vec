import importlib

import ast2vec.lazy_grpc as lazy_grpc

with lazy_grpc.masquerade():
    # All the stuff below does not really need grpc so we arrange the delayed import.

    from bblfsh.sdkversion import VERSION

    DESCRIPTOR = importlib.import_module(
        "bblfsh.gopkg.in.bblfsh.sdk.%s.uast.generated_pb2" % VERSION).DESCRIPTOR
    Node = importlib.import_module(
        "bblfsh.gopkg.in.bblfsh.sdk.%s.uast.generated_pb2" % VERSION).Node

    def _get_role_id(role_name):
        return DESCRIPTOR.enum_types_by_name["Role"].values_by_name[role_name].number

    SIMPLE_IDENTIFIER = _get_role_id("IDENTIFIER")
