import logging
import os
import sys
os.environ["PYSPARK_PYTHON"] = sys.executable

from pyspark.sql import SparkSession  # nopep8
from sourced.engine import Engine  # nopep8


def create_spark(session_name, kwargs):
    log = logging.getLogger("spark")
    log.info("Starting %s on %s", session_name, kwargs.spark)
    builder = SparkSession.builder.master(kwargs.spark).appName(session_name)
    builder = builder.config(
        "spark.jars.packages", ",".join(kwargs.package))
    builder = builder.config("spark.local.dir", kwargs.spark_local_dir)
    for cfg in kwargs.config:
        builder = builder.config(*cfg.split("=", 1))
    session = builder.getOrCreate()
    session.sparkContext.setLogLevel(logging._levelToName[log.getEffectiveLevel()])
    return session


def create_engine(session_name, repositories, kwargs):
    kwargs.config.append("spark.tech.sourced.bblfsh.grpc.host=" + kwargs.bblfsh)
    # TODO(vmarkovtsev): figure out why is this option needed
    kwargs.config.append("spark.tech.sourced.engine.cleanup.skip=true")
    kwargs.package.append("tech.sourced:engine:" + kwargs.engine)
    session = create_spark(session_name, kwargs)
    log = logging.getLogger("engine")
    log.info("Initializing on %s", repositories)
    engine = Engine(session, repositories)
    return engine
