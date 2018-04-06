import functools
import logging
import pip

from sourced.engine import Engine
from sourced.ml.utils import add_spark_args, create_spark, assemble_spark_config


class EngineConstants:
    """
    Constants for Engine usage.
    """
    class Columns:
        """
        Column names constants.
        """
        RepositoryId = "repository_id"
        Path = "path"
        BlobId = "blob_id"
        Uast = "uast"


class EngineDefault:
    """
    Default arguments for create_engine function and __main__
    """
    BBLFSH = "localhost"
    VERSION = {pkg.key: pkg.version for pkg in pip.get_installed_distributions()}["sourced-engine"]


def add_engine_args(my_parser, default_packages=None):
    add_spark_args(my_parser, default_packages=default_packages)
    my_parser.add_argument("--bblfsh", default=EngineDefault.BBLFSH,
                           help="Babelfish server's address.")
    my_parser.add_argument("--engine", default=EngineDefault.VERSION,
                           help="source{d} engine version.")
    my_parser.add_argument("--repository-format", default="siva",
                           help="Repository storage input format.")
    my_parser.add_argument("--explain", action="store_true",
                           help="Print the PySpark execution plans.")


def add_engine_dependencies(engine=EngineDefault.VERSION, config=None, packages=None):
    config.append("spark.tech.sourced.engine.cleanup.skip=true")
    packages.append("tech.sourced:engine:" + engine)


def add_bblfsh_dependencies(bblfsh, config=None):
    config.append("spark.tech.sourced.bblfsh.grpc.host=" + bblfsh)


def create_engine(session_name, repositories,
                  bblfsh=EngineDefault.BBLFSH,
                  engine=EngineDefault.VERSION,
                  config=None, packages=None, memory="",
                  repository_format="siva", **spark_kwargs):
    config, packages = assemble_spark_config(config=config, packages=packages, memory=memory)
    add_engine_dependencies(engine=engine, config=config, packages=packages)
    add_bblfsh_dependencies(bblfsh=bblfsh, config=config)
    session = create_spark(session_name, config=config, packages=packages, **spark_kwargs)
    log = logging.getLogger("engine")
    log.info("Initializing on %s", repositories)
    engine = Engine(session, repositories, repository_format)
    return engine


def pause(func):
    @functools.wraps(func)
    def wrapped_pause(cmdline_args, *args, **kwargs):
        try:
            return func(cmdline_args, *args, **kwargs)
        finally:
            if cmdline_args.pause:
                input("Press Enter to exit...")

    return wrapped_pause


def pipeline_graph(args, log, root):
    if hasattr(args, "graph") and args.graph:
        log.info("Dumping the graph to %s", args.graph)
        with open(args.graph, "w") as f:
            root.graph(stream=f)
