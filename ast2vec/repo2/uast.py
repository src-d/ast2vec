from ast2vec.repo2.base import Repo2Base, RepoTransformer, repos2_entry, repo2_entry
from ast2vec.uast import UASTModel


class Repo2UASTModel(Repo2Base):
    """
    Extract UASTs from repository
    """
    MODEL_CLASS = UASTModel

    def convert_uasts(self, file_uast_generator):
        uasts = []
        filenames = []

        for file_uast in file_uast_generator:
            uasts.append(file_uast.response)
            filenames.append(file_uast.filename)

        return filenames, uasts


class Repo2UASTModelTransformer(RepoTransformer):
    WORKER_CLASS = Repo2UASTModel

    def dependencies(self):
        return []

    def result_to_model_kwargs(self, result, url_or_path):
        filenames, uasts = result
        if len(filenames) == 0:
            raise ValueError('No need to store empty model.')
        return {"filenames": filenames, "uasts": uasts}


def repo2uast_entry(args):
    return repo2_entry(args, Repo2UASTModelTransformer)


def repos2uast_entry(args):
    return repos2_entry(args, Repo2UASTModelTransformer)
