from itertools import repeat
import logging
import multiprocessing
import os
from pathlib import Path
import threading
from typing import Union

from modelforge import Model
from modelforge.progress_bar import progress_bar

from ast2vec.pickleable_logger import PickleableLogger


class Model2Base(PickleableLogger):
    """
    Base class for model -> model conversions.
    """
    MODEL_FROM_CLASS = None
    MODEL_TO_CLASS = None

    def __init__(self, num_processes: int=multiprocessing.cpu_count(),
                 log_level: int=logging.DEBUG):
        """
        Initializes a new instance of Model2Base class.

        :param num_processes: The number of processes to execute for conversion.
        :param log_level: Logging verbosity level.
        """
        super(Model2Base, self).__init__(log_level=log_level)
        self.num_processes = num_processes

    def convert(self, srcdir: str, destdir: str, pattern: str="**/*.asdf") -> int:
        """
        Performs the model -> model conversion. Runs the conversions in a pool of processes.

        :param srcdir: The directory to scan for the models.
        :param destdir: The directory where to store the models. The directory structure is \
                        preserved.
        :param pattern: glob pattern for the files.
        :return: The number of converted files.
        """
        self._log.info("Scanning %s", srcdir)
        files = [str(p) for p in Path(srcdir).glob(pattern)]
        print(files)
        self._log.info("Found %d files", len(files))
        queue_in = multiprocessing.Manager().Queue()
        queue_out = multiprocessing.Manager().Queue(1)
        processes = [multiprocessing.Process(target=self._process_entry,
                                             args=(i, destdir, srcdir, queue_in, queue_out))
                     for i in range(self.num_processes)]
        for p in processes:
            p.start()
        for f in files:
            queue_in.put(f)
        for _ in processes:
            queue_in.put(None)
        failures = 0
        for _ in progress_bar(files, self._log, expected_size=len(files)):
            filename, ok = queue_out.get()
            if not ok:
                failures += 1
        for p in processes:
            p.join()
        self._log.info("Finished, %d failed files", failures)
        return len(files) - failures

    def convert_model(self, model: Model) -> Union[Model, None]:
        """
        This must be implemented in the child classes.

        :param model: The model instance to convert.
        :return: The converted model instance or None if it is not needed.
        """
        raise NotImplementedError

    def finalize(self, index: int, destdir: str):
        """
        Called for each worker in the end of the processing.

        :param index: Worker's index.
        :param destdir: The directory where to store the models.
        """
        pass

    def _process_entry(self, index, destdir, srcdir, queue_in, queue_out):
        while True:
            filename = queue_in.get()
            if filename is None:
                break
            try:
                model_from = self.MODEL_FROM_CLASS().load(filename)
                model_to = self.convert_model(model_from)
                if model_to is not None:
                    model_path = self._get_model_path(os.path.relpath(filename, srcdir))
                    model_path = os.path.join(destdir, model_path)
                    dirs = os.path.dirname(model_path)
                    if dirs:
                        os.makedirs(dirs, exist_ok=True)
                    model_to.save(model_path)
            except:
                self._log.exception("%s failed", filename)
                queue_out.put((filename, False))
            else:
                queue_out.put((filename, True))
        self.finalize(index, destdir)

    def _get_log_name(self):
        return "%s2%s" % (self.MODEL_FROM_CLASS.NAME, self.MODEL_TO_CLASS.NAME)

    def _get_model_path(self, path):
        """
        By default, we name the converted files exactly the same.

        :param path: The path relative to ``srcdir``.
        :return: The target path for the converted model.
        """
        return path
