from ast2vec.pickleable_logger import PickleableLogger  # nopep8


class Repo2FinalizerBase(PickleableLogger):
    def __call__(self, processed):
        raise NotImplementedError()


class Repo2Base(PickleableLogger):
    def __init__(self, engine, finalizer, languages=None, **kwargs):
        super().__init__(**kwargs)
        self.engine = engine
        self.finalizer = finalizer
        self.languages = languages or ("Java", "Python")

    def _get_log_name(self):
        return self.__class__.__name__

    def __getstate__(self):
        state = super().__getstate__()
        del state["engine"]
        del state["finalizer"]
        state["serialized"] = True
        return state

    def process_files(self, files=None):
        assert not getattr(self, "serialized", False)
        if files is None:
            files = self.engine.repositories.references.head_ref.files
        classified = files.classify_languages()
        lang_filter = classified.lang == self.languages[0]
        for lang in self.languages[1:]:
            lang_filter |= classified.lang == lang
        filtered_by_lang = classified.filter(lang_filter)
        uasts = filtered_by_lang.extract_uasts()

        process_uast = self.prepare_process_uast()
        processed = uasts.rdd.flatMap(process_uast)
        self.finalizer(processed)

    def process_uast(self, uast):
        raise NotImplementedError()
