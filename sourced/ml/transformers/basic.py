import logging
from typing import Union

from pyspark import StorageLevel, Row, RDD
from pyspark.sql import DataFrame

from sourced.ml.transformers.transformer import Transformer
from sourced.ml.transformers.uast2bag_features import Uast2BagFeatures
from sourced.ml.utils import assemble_spark_config, create_engine, create_spark, EngineConstants


class CsvSaver(Transformer):
    def __init__(self, output: str, **kwargs):
        super().__init__(**kwargs)
        self.output = output

    def __call__(self, head: RDD):
        self._log.info("Writing %s", self.output)
        return head.toDF() \
            .coalesce(1) \
            .write \
            .option("header", "true") \
            .mode("overwrite") \
            .csv(self.output)


class Rower(Transformer):
    def __init__(self, dicter, **kwargs):
        super().__init__(**kwargs)
        self.dicter = dicter

    def __call__(self, head: RDD):
        return head.map(lambda x: Row(**self.dicter(x)))


class Sampler(Transformer):
    """
    Wraps `sample()` function from pyspark Dataframe.
    """
    def __init__(self, with_replacement=False, fraction=0.05, seed=42, **kwargs):
        super().__init__(**kwargs)
        self.with_replacement = with_replacement
        self.fraction = fraction
        self.seed = seed

    def __call__(self, head: RDD):
        return head.sample(self.with_replacement, self.fraction, self.seed)


class Collector(Transformer):
    def __call__(self, head: RDD):
        return head.collect()


class First(Transformer):
    def __call__(self, head: RDD):
        return head.first()


class Identity(Transformer):
    def __call__(self, head: RDD):
        return head


class Cacher(Transformer):
    def __init__(self, persistence, **kwargs):
        super().__init__(**kwargs)
        self.persistence = getattr(StorageLevel, persistence)
        self.head = None
        self.trace = None

    def __getstate__(self):
        state = super().__getstate__()
        state["head"] = None
        state["trace"] = None
        return state

    def __call__(self, head: RDD):
        if self.head is None or self.trace != self.path():
            self.head = head.persist(self.persistence)
            self.trace = self.path()
        return self.head

    @staticmethod
    def maybe(persistence):
        if persistence is not None:
            return Cacher(persistence)
        else:
            return Identity()


class Ignition(Transformer):
    def __init__(self, engine, **kwargs):
        super().__init__(**kwargs)
        self.engine = engine

    def __getstate__(self):
        state = super().__getstate__()
        del state["engine"]
        return state

    def __call__(self, _):
        return self.engine


class HeadFiles(Transformer):
    def __call__(self, engine):
        return engine.repositories.references.head_ref.commits.tree_entries.blobs


class Counter(Transformer):
    def __init__(self, distinct=False, approximate=False, **kwargs):
        super().__init__(**kwargs)
        self.distinct = distinct
        self.approximate = approximate

    def __call__(self, head: RDD):
        if self.distinct and not self.approximate:
            head = head.distinct()
        if self.explained:
            self._log.info("toDebugString():\n%s", head.toDebugString().decode())
        if not self.approximate or not self.distinct:
            return head.count()
        return head.countApproxDistinct()


class LanguageSelector(Transformer):
    def __init__(self, languages: Union[list, tuple], **kwargs):
        super().__init__(**kwargs)
        self.languages = languages

    def __call__(self, files) -> DataFrame:
        classified = files.classify_languages()
        lang_filter = classified.lang == self.languages[0]
        for lang in self.languages[1:]:
            lang_filter |= classified.lang == lang
        filtered_by_lang = classified.filter(lang_filter)
        from pyspark.sql import functions
        return filtered_by_lang.where(functions.length(functions.col("content")) > 0)


class UastExtractor(Transformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, files: DataFrame) -> DataFrame:
        from pyspark.sql import functions
        return files.extract_uasts().where(functions.size(functions.col("uast")) > 0)


class FieldsSelector(Transformer):
    def __init__(self, fields: Union[list, tuple], **kwargs):
        super().__init__(**kwargs)
        self.fields = fields

    def __call__(self, df: DataFrame) -> DataFrame:
        res = df.select(self.fields)
        if self.explained:
            self._log.info("toDebugString():\n%s", res.rdd.toDebugString().decode())
        return res


class ParquetSaver(Transformer):
    def __init__(self, save_loc, **kwargs):
        super().__init__(**kwargs)
        self.save_loc = save_loc

    def __call__(self, df: DataFrame):
        if self.explained:
            self._log.info("toDebugString():\n%s", df.rdd.toDebugString().decode())
        df.write.parquet(self.save_loc)


class ParquetLoader(Transformer):
    def __init__(self, session, paths, **kwargs):
        super().__init__(**kwargs)
        self.session = session
        self.paths = paths

    def __getstate__(self):
        state = super().__getstate__()
        del state["session"]
        return state

    def __call__(self, _):
        if isinstance(self.paths, (list, tuple)):
            return self.session.read.parquet(*self.paths)
        if isinstance(self.paths, str):
            return self.session.read.parquet(self.paths)
        raise ValueError


class UastDeserializer(Transformer):
    def __setstate__(self, state):
        super().__setstate__(state)
        from bblfsh import Node
        self.parse_uast = Node.FromString

    def __call__(self, rows: RDD) -> RDD:
        return rows.flatMap(self.deserialize_uast)

    def deserialize_uast(self, row: Row):
        if not row[EngineConstants.Columns.Uast]:
            return
        row_dict = row.asDict()
        row_dict[EngineConstants.Columns.Uast] = []
        for i, uast in enumerate(row[EngineConstants.Columns.Uast]):
            try:
                row_dict[EngineConstants.Columns.Uast].append(self.parse_uast(uast))
            except:  # nopep8
                self._log.error("\nBabelfish Error: Failed to parse uast for document %s for uast "
                                "#%s" % (row[Uast2BagFeatures.Columns.document], i))
        yield Row(**row_dict)


def create_parquet_loader(session_name, repositories, config=None, memory="", packages=None,
                          **spark_kwargs):
    config, packages = assemble_spark_config(config=config, packages=packages, memory=memory)
    session = create_spark(session_name, config=config, packages=packages,
                           **spark_kwargs)
    log = logging.getLogger("parquet")
    log.info("Initializing on %s", repositories)
    parquet = ParquetLoader(session, repositories)
    return parquet


def create_uast_source(args, session_name, extract_uast=True, select=HeadFiles):
    if args.parquet:
        start_point = create_parquet_loader(session_name, **args.__dict__)
        root = start_point
    else:
        root = create_engine(session_name, **args.__dict__)
        start_point = Ignition(root, explain=args.explain) \
            .link(select()) \
            .link(LanguageSelector(languages=args.languages))
        if extract_uast:
            start_point = start_point.link(UastExtractor())
    return root, start_point
