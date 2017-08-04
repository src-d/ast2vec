from os import path

ID2VEC = "id2vec_1000.asdf"
DOCFREQ = "docfreq_1000.asdf"
BOW = "bow_1000.asdf"
NBOW = "nbow_1000.asdf"
COOCC = "coocc.asdf"
VOCCOOCC = "voccoocc.asdf"
UAST = "uast.asdf"

DATA_DIR_SOURCE = path.join(path.dirname(__file__), "source")
SOURCE_FILENAME = "test_example"
SOURCE = path.join(DATA_DIR_SOURCE, "%s.asdf" % SOURCE_FILENAME)
SOURCE_PY = path.join(DATA_DIR_SOURCE, "%s.py" % SOURCE_FILENAME)
SOURCE_UAST = path.join(DATA_DIR_SOURCE, "%s.uast" % SOURCE_FILENAME)

JOINBOWS = path.join(path.dirname(__file__), "merge_bows")
