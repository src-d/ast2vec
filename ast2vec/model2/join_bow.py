import multiprocessing
import tempfile
import os

import numpy

from ast2vec.bow import BOW, NBOW
from ast2vec.model2.base import Model2Base


class BowJoinerBase(Model2Base):
    def __init__(self, *args, **kwargs):
        super(BowJoinerBase, self).__init__(*args, **kwargs)
        self.joined = None

    def convert_model(self, model):
        if self.joined is None:
            self.joined = model
            self.joined._matrix.data = list(self.joined._matrix.data)
            self.joined._matrix.indices = list(self.joined._matrix.indices)
            self.joined._matrix.indptr = list(self.joined._matrix.indptr)
        else:
            if self.joined._matrix.shape != model._matrix.shape:
                self._log.warning("%s: matrix shape does not match, skipped", model._source)
                return None
            self.joined.repos.extend(model.repos)
            self.joined._matrix.data.extend(model._matrix.data)
            self.joined._matrix.indices.extend(model._matrix.indices)
            self.joined._matrix.indptr.extend(
                self.joined._matrix.indptr[-1] + model._matrix.indptr[1:])
        return None

    def finalize(self, index, destdir):
        if self.joined is None:
            return
        if destdir.endswith(".asdf"):
            name = destdir
        else:
            name = os.path.join(destdir, "%s%d.asdf" % (self.joined.meta["model"], index))
        self.joined._matrix.data = numpy.array(self.joined._matrix.data)
        self.joined._matrix.indices = numpy.array(self.joined._matrix.indices)
        self.joined._matrix.indptr = numpy.array(self.joined._matrix.indptr)
        self.joined._matrix._shape = (len(self.joined._matrix.indptr) - 1,
                                      self.joined._matrix.shape[1])
        self.joined.save(output=name, deps=self.joined.meta["dependencies"])


class BowJoiner(BowJoinerBase):
    MODEL_FROM_CLASS = BOW
    MODEL_TO_CLASS = BOW


class NbowJoiner(BowJoinerBase):
    MODEL_FROM_CLASS = NBOW
    MODEL_TO_CLASS = NBOW


def joinbow_entry(args):
    processes = args.processes or multiprocessing.cpu_count()
    if args.nbow:
        joiner = NbowJoiner(num_processes=processes)
    else:
        joiner = BowJoiner(num_processes=processes)
    with tempfile.TemporaryDirectory(dir=args.tmpdir, prefix="joinbow") as tmpdir:
        joiner.convert(args.input, tmpdir, pattern=args.filter)
        if args.nbow:
            joiner = NbowJoiner(num_processes=1)
        else:
            joiner = BowJoiner(num_processes=1)
        joiner.convert(tmpdir, args.output,
                       pattern="%s*.asdf" % (NBOW.NAME if args.nbow else BOW.NAME))
