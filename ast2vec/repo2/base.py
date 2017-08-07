from collections import namedtuple
from itertools import repeat
import logging
import multiprocessing
import os
from queue import Queue
import shutil
import subprocess
import tempfile
import threading

from bblfsh import BblfshClient
from bblfsh.launcher import ensure_bblfsh_is_running
from google.protobuf.message import DecodeError
from modelforge.progress_bar import progress_bar

from ast2vec.cloning import RepoCloner
from ast2vec.pickleable_logger import PickleableLogger
from ast2vec.token_parser import TokenParser
from ast2vec import resolve_symlink

GeneratorResponse = namedtuple("GeneratorResponse",
                               ["filepath", "filename", "response"])


class Repo2Base(PickleableLogger):
    """
    Base class for repsitory features extraction. Abstracts from
    `Babelfish <https://doc.bblf.sh/>`_ and source code identifier processing.
    """
    MODEL_CLASS = None  #: Must be defined in the children.
    DEFAULT_BBLFSH_TIMEOUT = 10  #: Longer requests are dropped.
    MAX_FILE_SIZE = 200000

    def __init__(self, tempdir=None, linguist=None, log_level=logging.INFO,
                 bblfsh_endpoint=None, timeout=DEFAULT_BBLFSH_TIMEOUT):
        super(Repo2Base, self).__init__(log_level=log_level)
        self._tempdir = tempdir
        self._cloner = RepoCloner(redownload=True, log_level=log_level)
        self._cloner.find_linguist(linguist)
        self._bblfsh = [BblfshClient(bblfsh_endpoint or "0.0.0.0:9432")
                        for _ in range(multiprocessing.cpu_count())]
        self._timeout = timeout
        self._token_parser = TokenParser()

    def convert_repository(self, url_or_path):
        """
        Queries bblfsh for the UASTs and produces smth useful from them.

        :param url_or_path: File system path to the repository or a URL to clone.
        :return: Some object(s) which are returned from convert_uasts().
        """
        temp = not os.path.exists(url_or_path)
        if temp:
            target_dir = tempfile.mkdtemp(prefix="repo2-", dir=self._tempdir)
            target_dir = self._cloner.clone_repo(url_or_path, ignore=False, target_dir=target_dir)
        else:
            target_dir = url_or_path
        try:
            classified = self._cloner.classify_repo(target_dir)
            self._log.info("Fetching and processing UASTs...")

            def file_uast_generator():
                queue_in = Queue()
                queue_out = Queue()

                def thread_loop(thread_index):
                    while True:
                        task = queue_in.get()
                        if task is None:
                            break
                        try:
                            dirname, filename, language = task
                            filepath = os.path.join(dirname, filename)

                            try:
                                # Resolve symlink
                                filepath = resolve_symlink.resolve_symlink(filepath)
                            except resolve_symlink.DanglingSymlinkError as e:
                                self._log.warning(*e.args)
                                queue_out.put_nowait(None)
                                continue

                            size = os.stat(filepath).st_size
                            if size > self.MAX_FILE_SIZE:
                                self._log.warning("%s is too big - %d", filepath, size)
                                queue_out.put_nowait(None)
                                continue

                            response = self._bblfsh_parse(thread_index, filepath, language)
                            if response is None:
                                self._log.warning("bblfsh timed out on %s", filepath)
                                queue_out.put_nowait(None)
                                continue

                            queue_out.put_nowait(GeneratorResponse(filepath=filepath,
                                                                   filename=filename,
                                                                   response=response))
                        except:
                            self._log.exception(
                                "Error while processing %s", task)
                            queue_out.put_nowait(None)

                pool = [threading.Thread(target=thread_loop, args=(i,),
                                         name="%s@%d" % (url_or_path, i))
                        for i in range(multiprocessing.cpu_count())]
                for thread in pool:
                    thread.start()
                tasks = 0
                empty = True
                lang_list = ("Python", "Java")
                for lang, files in classified.items():
                    # FIXME(vmarkovtsev): remove this hardcode when https://github.com/bblfsh/server/issues/28 is resolved # nopep8
                    if lang not in lang_list:
                        continue
                    for f in files:
                        tasks += 1
                        empty = False
                        queue_in.put_nowait((target_dir, f, lang))
                report_interval = max(1, tasks // 100)
                for _ in pool:
                    queue_in.put_nowait(None)
                while tasks > 0:
                    result = queue_out.get()
                    if result is not None:
                        yield result
                    tasks -= 1
                    if tasks % report_interval == 0:
                        self._log.info("%s pending tasks: %d", url_or_path, tasks)
                for thread in pool:
                    thread.join()

                if empty:
                    self._log.warning("No files were processed for %s", url_or_path)

            return self.convert_uasts(file_uast_generator())
        finally:
            if temp:
                shutil.rmtree(target_dir)

    def convert_uast(self, uast):
        return self.convert_uasts([uast])

    def convert_uasts(self, file_uast_generator):
        raise NotImplementedError()

    def _bblfsh_parse(self, thread_index, filepath, language):
        try:
            return self._bblfsh[thread_index].parse(
                filepath, language=language, timeout=self._timeout)
        except DecodeError as e:
            msg = "bblfsh raised an DecodeError exception. Probably your protobuf is <= v3.3.2 " \
                  "and you hit https://github.com/bblfsh/server/issues/59#issuecomment-318125752"
            self._log.warning(msg)
            raise e from None

    def _get_log_name(self):
        return "repo2" + self.MODEL_CLASS.NAME


class Transformer(PickleableLogger):
    """
    Base class for transformers
    """

    def transform(self, *args, **kwargs):
        return NotImplementedError()


class RepoTransformer(Transformer):
    WORKER_CLASS = None
    DEFAULT_NUM_PROCESSES = 2

    def __init__(self, num_processes=DEFAULT_NUM_PROCESSES, **args):
        super(RepoTransformer, self).__init__()
        self._args = args
        self._num_processes = num_processes

    @property
    def num_processes(self):
        return self._num_processes

    @num_processes.setter
    def num_processes(self, value):
        if not isinstance(value, int):
            raise TypeError("num_processes must be an integer")
        self._num_processes = value

    @classmethod
    def process_entry(cls, url_or_path, args, outdir, queue):
        """
        Invokes process_repo() in a separate process. The reason we do this is that grpc
        starts hanging background threads for every channel which poll(). Those threads
        do not exit when the channel is destroyed. It is fine for a single repository, but
        quickly hits the system limit in case of many.

        This method is intended for the batch processing.

        :param url_or_path: File system path or a URL to clone.
        :param args: :class:`dict`-like container with the arguments to cls().
        :param outdir: The output directory.
        :param queue: :class:`multiprocessing.Queue` to report the status.
        :return:
        """
        pid = os.fork()
        if pid == 0:
            outfile = cls.prepare_filename(url_or_path, outdir)
            status = cls(**args).process_repo(url_or_path, outfile)
            import sys
            sys.exit(status)
        else:
            _, status = os.waitpid(pid, 0)
            queue.put((url_or_path, status))

    @classmethod
    def prepare_filename(cls, repo, output):
        """
        Remove prefixes from the repo name, so later it can be used to create
        file for each repository + replace slashes ("/") with ampersands ("&").

        :param repo: name of repository
        :param output: output directory
        :return: converted repository name (removed "https://", etc.)
        """
        repo_name = repo
        prefixes = ["https://", "http://", "git://", "ssh://"]
        for prefix in prefixes:
            if repo.startswith(prefix):
                repo_name = repo_name[len(prefix):]
                break
        postfixes = "\n/\\.git"
        repo_name.rstrip(postfixes)
        for postfix in postfixes:
            if repo.endswith(postfix):
                repo_name = repo_name[:-len(postfix)]

        outfile = os.path.join(output, "%s_%s.asdf" % (
            cls.WORKER_CLASS.MODEL_CLASS.NAME, repo_name.replace("/", "&")))
        return outfile

    def process_repo(self, url_or_path, output) -> bool:
        """
        Pipeline for a single repository:

        1. Initialize the implementation class instance.
        2. Use it to convert the repository to a model.
        3. Save the result on disk.

        :param url_or_path: Repository URL or file system path.
        :param output: Path to file where to store the result.
        :return: True if the operation was successful; otherwise, False.
        """
        repo2 = self.WORKER_CLASS(**self._args)
        try:
            result = repo2.convert_repository(url_or_path)
            for proto in ("https://", "http://", "git://", "ssh://"):
                if url_or_path.startswith(proto):
                    url_or_path = url_or_path.replace(proto, "")
            model = self.WORKER_CLASS.MODEL_CLASS()
            model.construct(**self.result_to_model_kwargs(result, url_or_path))
            model.save(output, deps=self.dependencies())
            return True
        except subprocess.CalledProcessError as e:
            self._log.error("Failed to clone %s: %s", url_or_path, e)
            return False
        except:
            self._log.exception(
                "Unhandled error in %s.process_repo() at %s." % (
                    type(self).__name__, url_or_path))
            return False

    def transform(self, repos, output, num_processes=None):
        """
        Converts repositories to models and saves them to the output directory.

        :param repos: "repos" is the list of repository URLs or paths or \
                  files with repository URLS or paths.
        :param output: "output" is the output directory where to store the \
                        results.
        :param num_processes: number of processes to use, if negative - use all \
               CPUs.
        :return: None
        """
        if num_processes is None:
            num_processes = self.num_processes
        if num_processes < 0:
            num_processes = multiprocessing.cpu_count()

        inputs = []

        if isinstance(repos, str):
            repos = [repos]

        for repo in repos:
            # check if it's a text file
            if os.path.isfile(repo):
                with open(repo) as f:
                    inputs.extend(l.strip() for l in f)
            else:
                inputs.append(repo)

        os.makedirs(output, exist_ok=True)

        queue = multiprocessing.Manager().Queue(1)

        def process_repos():
            with multiprocessing.Pool(processes=num_processes) as pool:
                pool.starmap(type(self).process_entry,
                             zip(inputs, repeat(self._args), repeat(output), repeat(queue)))

        mpthread = threading.Thread(target=process_repos)
        mpthread.start()
        failures = 0
        for _ in progress_bar(inputs, self._log, expected_size=len(inputs)):
            repo, ok = queue.get()
            if not ok:
                failures += 1
        mpthread.join()
        self._log.info("Finished, %d failed repos", failures)
        return len(inputs) - failures

    def _get_log_name(self):
        return self.WORKER_CLASS.MODEL_CLASS.NAME + "_transformer"

    def dependencies(self) -> list:
        """
        Returns the list of parent models which were used to generate the target one.
        """
        raise NotImplementedError

    def result_to_model_kwargs(self, result, url_or_path: str) -> dict:
        """
        Converts the "result" object from parse_uasts() to WORKER_CLASS.MODEL_CLASS.construct()
        keyword arguments.

        :param result: The object returned from parse_uasts().
        :param url_or_path: The repository's source.
        :return: :class:`dict` with the required items to construct the model.
        """
        raise NotImplementedError


def ensure_bblfsh_is_running_noexc():
    """
    Launches the Babelfish server, if it is possible and needed.

    :return: None
    """
    try:
        ensure_bblfsh_is_running()
    except:
        log = logging.getLogger("bblfsh")
        message = "Failed to ensure that the Babelfish server is running."
        if log.isEnabledFor(logging.DEBUG):
            log.exception(message)
        else:
            log.warning(message)


def _sanitize_kwargs(args, *blacklist):
    payload_args = getattr(args, "__dict__", args).copy()
    blacklist += ("output", "command", "handler")
    for arg in blacklist:
        if arg in payload_args:
            del payload_args[arg]
    return payload_args


def repo2_entry(args, payload_class):
    """
    Invokes payload_class(\*\*args).process_repo() on the specified repository.

    :param args: :class:`argparse.Namespace` with "repository" and "output". \
                 "repository" is a file system path or a URL. "output" is the path \
                 to the file with the resulting model.
    :param payload_class: :class:`Transformer` inheritor to call.
    :return: None
    """
    ensure_bblfsh_is_running_noexc()
    payload_args = _sanitize_kwargs(args, "repository")
    payload_class(**payload_args).process_repo(args.repository, args.output)


def repos2_entry(args, payload_class):
    """
    Invokes payload_class(\*\*args).transform() for every repository in parallel processes.

    :param args: :class:`argparse.Namespace` with "input" and "output". \
                 "input" is the list of repository URLs or paths or files \
                 with repository URLS or paths. "output" is the output \
                 directory where to store the results.
    :param payload_class: :class:`Transformer` inheritor to call.
    :return: None
    """
    ensure_bblfsh_is_running_noexc()
    payload_args = _sanitize_kwargs(args, "input")
    payload_class(**payload_args).transform(args.input, args.output)
