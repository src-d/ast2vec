import operator
import os
from typing import NamedTuple, Generator
import yaml

import pygments
from pygments.formatter import Formatter
from pygments.lexers import get_lexer_by_name, ClassNotFound
from pyspark import RDD, Row
from pyspark.sql import DataFrame, functions

from sourced.ml.algorithms import TokenParser
from sourced.ml.transformers import Transformer
from sourced.ml.utils import EngineConstants


class ContentToIdentifiers(Transformer):

    class FormatterProxy(Formatter):
        name = "Proxy"
        aliases = ["proxy"]
        filenames = []

        def __init__(self, **options):
            super(ContentToIdentifiers.FormatterProxy, self).__init__(**options)
            self.callback = options["callback"]

        def format(self, tokensource, outfile):
            self.callback(tokensource)

    def __init__(self, split: bool, **kwargs):
        super().__init__(**kwargs)
        Columns = NamedTuple("Columns", [("repo_id", str), ("file_id", str)])
        self.column_names = Columns(repo_id=EngineConstants.Columns.RepositoryId,
                                    file_id=EngineConstants.Columns.Path)
        self.linguist2pygments = self.build_mapping()
        self.split = split

    def __call__(self, rows: DataFrame):
        rdd = rows .where(functions.length(functions.col("content")) > 0).rdd
        return rdd.flatMap(self.process_row)

    def process_row(self, row: Row):
        self.names = []
        repo_id = getattr(row, self.column_names.repo_id)
        file_id = getattr(row, self.column_names.file_id)
        path = os.path.join(repo_id, file_id)
        code = row.content
        for i in (0, 1):
            try:
                lexer = get_lexer_by_name(self.linguist2pygments[row.lang][i])
                pygments.highlight(code, lexer, self.FormatterProxy(callback=self.process_tokens))
                break
            except (KeyError, IndexError, ClassNotFound, UnicodeDecodeError) as e:
                continue
        for token in self.names:
            yield token, (repo_id, path)

    def process_tokens(self, tokens: Generator):
        """
        Filter tokens of type "Name" and which are splittable
        according to :class: 'TokenParser' rules
        """
        for _type, token in tokens:
            if _type[0] == "Name":
                if self.split:
                    if sum(1 for _ in TokenParser(min_split_length=1).split(token)) > 1:
                        self.names.append(token)
                else:
                    self.names.append(token)

    @staticmethod
    def build_mapping():
        """
        Builds the mapping between linguist languages and pygments names for lexers.
        """
        linguist2pygments = {}
        with open(os.path.join(os.path.dirname(__file__), "languages.yml")) as f:
            all_languages = yaml.load(f)

        linguist_langs = {}
        for lang, specs in all_languages.items():
            if specs["type"] == "programming":
                linguist_langs[lang] = (set(specs.get("aliases", []) + [lang.lower()]), specs)

        pygments_langs = set()
        for lexer in pygments.lexers.LEXERS.values():
            lang_declensions = [lang.lower() for lang in lexer[2]] + [lexer[1].lower()]
            pygments_langs |= set(lang_declensions)

        for lang in linguist_langs:
            lang_names = linguist_langs.get(lang, (set(),))[0]
            inter = list(lang_names.intersection(pygments_langs))
            if inter:
                linguist2pygments[lang] = inter
        return linguist2pygments


class IdentifiersToDataset(Transformer):

    def __init__(self, idfreq: bool, **kwargs):
        super().__init__(**kwargs)
        self.idfreq = idfreq

    def __call__(self, rows: RDD):
        if self.idfreq:
            return self.process_stats(rows)
        else:
            return self.process(rows)

    def reduce_rows(self, rows: RDD):
        return rows \
            .map(lambda x: (x[0], 1)) \
            .reduceByKey(operator.add)

    def process_stats(self, rows: RDD):
        """
        Process rows to gather identifier frequencies.
        num_repos is the number of repositories where the identifier appears in.
        num_files is the number of files where the identifier appears in.
        num_occ is the total number of occurences of the identifier.
        """
        list_RDDs = []
        for i in (0, 1):
            # initial structure of x: (identifier, (repositoryId, filepath))
            freq_processed = rows.map(lambda x: (x[0], x[1][i])).distinct()
            list_RDDs.append(self.reduce_rows(freq_processed))
        list_RDDs.append(self.reduce_rows(rows))
        return rows \
            .context.union(list_RDDs) \
            .groupByKey() \
            .mapValues(list) \
            .map(lambda x: Row(
                    token=x[0],
                    token_split=" ".join(TokenParser(min_split_length=1).split(x[0])),
                    num_repos=x[1][0],
                    num_files=x[1][1],
                    num_occ=x[1][2]))

    def process(self, rows: RDD):
        """
        Process rows to gather identifier without frequencies.
        """
        return rows \
            .keys() \
            .distinct() \
            .map(lambda x: Row(
                    token=x,
                    token_split=" ".join(TokenParser(min_split_length=1).split(x))))
