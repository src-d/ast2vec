from typing import Union

from pyspark import StorageLevel, Row

from sourced.ml.transformers.transformer import Transformer


class Collector(Transformer):
    def __call__(self, head):
        return head.collect()


class First(Transformer):
    def __call__(self, head):
        return head.first()


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

    def __call__(self, head):
        if self.head is None or self.trace != self.path():
            self.head = head.persist(self.persistence)
            self.trace = self.path()
        return self.head


class Engine(Transformer):
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
        return engine.repositories.references.head_ref.commits.first_reference_commit \
            .tree_entries.blobs


class UastExtractor(Transformer):
    def __init__(self, languages: Union[list, tuple], **kwargs):
        super().__init__(**kwargs)
        self.languages = languages

    def __call__(self, files):
        files = files.dropDuplicates(("blob_id",)).filter("is_binary = 'false'")
        classified = files.classify_languages()
        lang_filter = classified.lang == self.languages[0]
        for lang in self.languages[1:]:
            lang_filter |= classified.lang == lang
        filtered_by_lang = classified.filter(lang_filter)
        uasts = filtered_by_lang.extract_uasts()
        return uasts


class FieldsSelector(Transformer):
    def __init__(self, fields: Union[list, tuple], **kwargs):
        super().__init__(**kwargs)
        self.fields = fields

    def __call__(self, df):
        return df.select(self.fields)


class ParquetSaver(Transformer):
    def __init__(self, save_loc, **kwargs):
        super().__init__(**kwargs)
        self.save_loc = save_loc

    def __call__(self, df):
        df.write.parquet(self.save_loc)


class ParquetLoader(Transformer):
    def __init__(self, session, **kwargs):
        super().__init__(**kwargs)
        self.session = session

    def __call__(self, df):
        return self.session.read.parquet(self.save_loc)


class UastDeserializer(Transformer):
    def __setstate__(self, state):
        super().__setstate__(state)
        from bblfsh import Node
        self.parse_uast = Node.FromString

    def __call__(self, rows):
        return rows.rdd.flatMap(self.deserialize_uast)

    def deserialize_uast(self, row):
        if not row.uast:
            return
        row_dict = row.asDict()
        row_dict["uast"] = self.parse_uast(row.uast[0])
        yield Row(**row_dict)
