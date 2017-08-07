from ast2vec.repo2.base import Repo2Base
from ast2vec.repo2.base import RepoTransformer
from ast2vec.source import Source


class Repo2Source(Repo2Base):
    """
    Extract source code and uast of repository of certain languages
    """
    MODEL_CLASS = Source

    def __init__(self, *args, **kwargs):
        super(Repo2Source, self).__init__(*args, **kwargs)
        self._uast_only = False

    def convert_uasts(self, file_uast_generator):
        sources = []
        uasts = []
        filenames = []

        for file_uast in file_uast_generator:
            source = self._get_source(file_uast.filepath)
            if source is None:
                continue
            sources.append(source)
            uasts.append(file_uast.response)
            filenames.append(file_uast.filename)

        if not len(sources) == len(uasts) == len(filenames):
            raise ValueError("Length of src_codes({}), uasts({}) and filenames({}) are not equal".
                             format(len(sources), len(uasts), len(filenames)))

        return filenames, sources, uasts

    def _get_source(self, filename):
        try:
            with open(filename, "r", encoding="utf8") as f:
                return f.read()
        except UnicodeDecodeError as e:
            self._log.warning('Skipping file %s.\n\tUnicodeDecodeError: %s', filename, e)


class Repo2SourceTransformer(RepoTransformer):
    WORKER_CLASS = Repo2Source

    def dependencies(self):
        return []

    def result_to_model_kwargs(self, result, url_or_path):
        filenames, src_codes, uasts = result
        if len(filenames) == 0:
            raise ValueError("The model is empty")
        return {"filenames": filenames, "sources": src_codes, "uasts": uasts}
