from os.path import dirname, join

_root = dirname(__file__)
_models_path = join(_root, "asdf")

ID2VEC = join(_models_path, "id2vec_1000.asdf")
DOCFREQ = join(_models_path, "docfreq_1000.asdf")
QUANTLEVELS = join(_models_path, "quant.asdf")
BOW = join(_models_path, "bow.asdf")
COOCC = join(_models_path, "coocc.asdf")
COOCC_DF = join(_models_path, "coocc_df.asdf")
UAST = join(_models_path, "uast.asdf")
TOPICS = join(_models_path, "topics.asdf")

DATA_DIR_SOURCE = join(_root, "source")
SOURCE_FILENAME = "example"
SOURCE = join(DATA_DIR_SOURCE, "%s.asdf" % SOURCE_FILENAME)
SOURCE_PY = join(DATA_DIR_SOURCE, "%s.py" % SOURCE_FILENAME)

TOPICS_SRC = "topics_readable.txt"

PARQUET_DIR = join(_root, "parquet")
