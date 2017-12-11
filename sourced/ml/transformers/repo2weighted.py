from sourced.ml.transformers.transformer import Transformer


class Repo2WeightedSet(Transformer):
    def __init__(self, extractors, **kwargs):
        super().__init__(**kwargs)
        self.extractors = extractors

    def __call__(self, rows):
        return rows.map(self.process_row)

    def process_row(self, row):
        bag = {}
        for extractor in self.extractors:
            bag.update(extractor.extract(row.uast))
        return row.blob_id, bag