import importlib

import sourced.ml.lazy_grpc as lazy_grpc

with lazy_grpc.masquerade():
    # All the stuff below does not really need grpc so we arrange the delayed import.

    from bblfsh.sdkversion import VERSION

    Node = importlib.import_module(
        "bblfsh.gopkg.in.bblfsh.sdk.%s.uast.generated_pb2" % VERSION).Node

    _ROLE = importlib.import_module(
        "bblfsh.gopkg.in.bblfsh.sdk.%s.uast.generated_pb2" % VERSION)._ROLE

    class Roles:
        """
        Contain babelfish roles ids.
        """
        pass

    for desc in _ROLE.values:
        setattr(Roles, desc.name, desc.index)
    del _ROLE
