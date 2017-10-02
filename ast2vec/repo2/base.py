from collections import namedtuple
from itertools import accumulate, repeat
import logging
from math import isnan
import multiprocessing
import os
from queue import Queue
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
from typing import Union
from time import time

from google.protobuf.message import DecodeError
from modelforge.logs import setup_logging
from modelforge.progress_bar import progress_bar
import netifaces

import ast2vec.lazy_grpc as lazy_grpc
with lazy_grpc.masquerade():
    from bblfsh import BblfshClient
    from bblfsh.launcher import ensure_bblfsh_is_running

from ast2vec.cloning import RepoCloner  # nopep8
from ast2vec.pickleable_logger import PickleableLogger  # nopep8
from ast2vec import resolve_symlink  # nopep8

GeneratorResponse = namedtuple("GeneratorResponse", ["filepath", "filename", "response"])
DEFAULT_BBLFSH_ENDPOINTS = ["0.0.0.0:9432"] + [
    "%s:9432" % gw[0] for gw in sorted(netifaces.gateways()["default"].values())]
DEFAULT_BBLFSH_TIMEOUT = 20  # Longer requests are dropped.
DEFAULT_PARSE_LOG_FILENAME = "bblfsh_parse_files.log"  # Filename for additional debug logs

lock = threading.Lock()


def resolve_parse_log_filename():
    parse_log_filename = os.getenv("AST2VEC_PARSE_LOG")
    if parse_log_filename:
        return parse_log_filename
    return None


class BblfshFailedError(Exception):
    """
    Raised when we receive errors from bblfsh server.
    """
    pass


class Repo2Base(PickleableLogger):
    """
    Base class for repsitory features extraction. Abstracts from
    `Babelfish <https://doc.bblf.sh/>`_ and source code identifier processing.
    """
    MODEL_CLASS = None  #: Must be defined in the children.
    DEFAULT_BBLFSH_RAISE_ERRORS = False  # Set `True` to fail on bblfsh errors.
    DEFAULT_OVERWRITE_EXISTING = True
    DEFAULT_THREAD_COUNT = multiprocessing.cpu_count()
    MAX_FILE_SIZE = 200000
    PARSE_LOG_FILENAME = resolve_parse_log_filename()

    def __init__(self, tempdir=None, linguist=None, log_level=logging.INFO,
                 bblfsh_endpoint=None, timeout=None,
                 threads=DEFAULT_THREAD_COUNT,
                 overwrite_existing=DEFAULT_OVERWRITE_EXISTING,
                 bblfsh_raise_errors=DEFAULT_BBLFSH_RAISE_ERRORS):
        """
        Initializer of Repo2Base class
        :param tempdir: If you clone repositories they will be stored in tempdir
        :param linguist: Path to linguist.
        :param log_level: Log level of Repo2Base
        :param bblfsh_endpoint: bblfsh server endpoint
        :param timeout: timeout for bblfsh
        :param overwrite_existing: Rewrite existing models or skip them
        :param bblfsh_raise_errors: If `True` raises errors when you get errors in bblfsh responses
        """
        super().__init__(log_level=log_level)
        self.tempdir = tempdir
        self._cloner = RepoCloner(redownload=True, log_level=log_level)
        self._cloner.find_linguist(linguist)
        self._bblfsh_endpoint = resolve_bblfsh_endpoint(bblfsh_endpoint)
        self._bblfsh_raise_errors = bblfsh_raise_errors
        self._overwrite_existing = overwrite_existing
        self.timeout = resolve_bblfsh_timeout(timeout)
        self.threads = threads
        self._lines = 0
        self._time = 0
        self._parse_log_filepath = resolve_parse_log_filename()
        if self.PARSE_LOG_FILENAME:
            with open(self.PARSE_LOG_FILENAME, "w") as f:
                f.write("Thread, lines, time, speed, status, path")

    @property
    def tempdir(self):
        """
        The temporary directory where to clone the repositories.
        """
        return self._tempdir

    @tempdir.setter
    def tempdir(self, value: Union[str, None]):
        if value is not None and not os.path.isdir(value):
            raise ValueError("Path does not exist: %s" % value)
        self._tempdir = value

    @property
    def bblfsh_endpoint(self):
        return self._bblfsh_endpoint

    @property
    def timeout(self):
        """
        Babelfish server timeout.
        """
        return self._timeout

    @timeout.setter
    def timeout(self, value: Union[int, float, None]):
        if value is not None:
            if not isinstance(value, (int, float)):
                raise TypeError("timeout must be an int/float, got %s" % type(value))
            if value <= 0:
                raise ValueError("timeout must be positive, got %s" % value)
        self._timeout = value

    @property
    def threads(self):
        """
        The number of threads in the repository -> UASTs extraction process.
        """
        return self._threads

    @threads.setter
    def threads(self, value: int):
        if not isinstance(value, int):
            raise TypeError("threads must be an integer - got %s" % type(value))
        if value < 1:
            raise ValueError("threads must be greater than or equal to 1 - got %d" % value)
        self._threads = value
        self._bblfsh = [BblfshClient(self.bblfsh_endpoint) for _ in range(value)]

    @property
    def overwrite_existing(self):
        return self._overwrite_existing

    @overwrite_existing.setter
    def overwrite_existing(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError("overwrite_existing must be a boolean - got %s" % type(value))
        self._overwrite_existing = value

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
            file_uast_generator = self._file_uast_generator(classified, target_dir, url_or_path)
            return self.convert_uasts(file_uast_generator)
        except BblfshFailedError as e:
            self._log.error(e)
        finally:
            if temp:
                shutil.rmtree(target_dir)

    def convert_uast(self, uast):
        return self.convert_uasts([uast])

    def convert_uasts(self, file_uast_generator):
        raise NotImplementedError()

    def _bblfsh_parse(self, thread_index, filepath, language):
        from grpc import RpcError
        status = "ERR"
        start_time = time()
        if self.PARSE_LOG_FILENAME:
            lines = lines_count(filepath, self._log)
        try:
            res = self._bblfsh[thread_index].parse(
                filepath, language=language, timeout=self._timeout)
            status = "OK"
            return res
        except DecodeError as e:
            msg = "bblfsh: DecodeError on %s: %s\nYour protobuf may be <= v3.3.2 " \
                  "and you hit https://github.com/bblfsh/server/issues/59#issuecomment-318125752"
            self._log.error(msg, e)
        except RpcError as e:
            self._log.error("bblfsh: RpcError on %s: %s", filepath, e)
            raise e from None
        finally:
            if self.PARSE_LOG_FILENAME:
                self._write_debug_logs(thread_index, filepath, status, lines, start_time)

    def _write_debug_logs(self, thread_index, filepath, status, lines, start_time):
        time_spent = time() - start_time
        if not isnan(lines):
            self._lines += lines
            self._time += time_spent
        if self.PARSE_LOG_FILENAME:
            with lock:
                with open(self.PARSE_LOG_FILENAME, "a") as f:
                    f.write("% 2d, % 5d, % 6.2f, % 8.2f, % 3s, %s\n" %
                            (thread_index, lines, time_spent, lines / time_spent, status,
                             os.path.relpath(filepath, os.getcwd())))

    def _file_uast_generator(self, classified, target_dir, url_or_path):
        queue_in = Queue()
        queue_out = Queue()
        errors = []
        errors_lock = threading.Lock()

        def thread_loop(thread_index):
            while True:
                task = queue_in.get()
                if task is None or errors:
                    break
                try:
                    dirname, filename, language = task
                    filepath = os.path.join(dirname, filename)

                    def skip(*args, addmsg=""):
                        self._log.warning("%s was skipped" + addmsg, filepath, *args)
                        queue_out.put_nowait(None)

                    try:
                        filepath = resolve_symlink.resolve_symlink(filepath)
                    except resolve_symlink.DanglingSymlinkError as e:
                        self._log.error(*e.args)
                        skip()
                        continue

                    size = os.stat(filepath).st_size
                    if size > self.MAX_FILE_SIZE:
                        skip(size, addmsg=": it is too big - %d bytes")
                        continue

                    response = self._bblfsh_parse(thread_index, filepath, language)
                    if response is None:
                        skip()
                        continue
                    if response.errors and self._bblfsh_raise_errors:
                        with errors_lock:
                            errors.extend(response.errors)
                        skip()
                        break

                    queue_out.put_nowait(GeneratorResponse(filepath=filepath,
                                                           filename=filename,
                                                           response=response))
                except:
                    self._log.exception("While processing %s", task)
                    queue_out.put_nowait(None)

        pool = [threading.Thread(target=thread_loop, args=(i,),
                                 name="%s@%d" % (url_or_path, i))
                for i in range(self.threads)]
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
        while tasks > 0 and not errors:
            result = queue_out.get()
            if result is not None:
                yield result
            tasks -= 1
            if tasks % report_interval == 0:
                self._log.info("%s pending tasks: %d", url_or_path, tasks)
        for thread in pool:
            thread.join()

        if errors:
            raise BblfshFailedError(
                "Received errors: %s", ", ".join(errors))
        if empty:
            self._log.warning("No files were processed for %s", url_or_path)
        if self.PARSE_LOG_FILENAME:
            self._log.debug("Average bblfsh speed % 6.2f line/sec for %s" %
                            (self._lines / self._time, url_or_path))

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
    DEFAULT_ORGANIZE_FILES = 0

    def __init__(self, log_level=logging.INFO, num_processes=DEFAULT_NUM_PROCESSES,
                 organize_files=DEFAULT_ORGANIZE_FILES, **kwargs):
        """
        Base class for transformers from repository to WORKER_CLASS model
        :param log_level: Logging level of this instance.
        :param num_processes: Number of parallel processes to transform
        :param organize_files: Perform alphabetical directory indexing of provided level.
            Expand output path by subfolders using the first n characters of repository,
            for example for "organize_files=2" file ababa is saved to /a/ab/ababa, bcoasa
            is saved to /b/bc/bcoasa, etc.
        :param args: arguments for WORKER_CLASS model initialization
        """
        super().__init__(log_level=log_level)
        self._args = kwargs
        self._args["log_level"] = log_level
        self._num_processes = num_processes
        self._organize_files = organize_files
        self._lines = 0
        self._time = 0

    @property
    def num_processes(self):
        return self._num_processes

    @num_processes.setter
    def num_processes(self, value):
        if not isinstance(value, int):
            raise TypeError("num_processes must be an integer - got %s" % type(value))
        if value < 1:
            raise ValueError("num_processes must be greater than or equal to 1 - got %d" % value)
        self._num_processes = value

    @classmethod
    def process_entry(cls, url_or_path: str, args: dict, outdir: str,
                      queue: multiprocessing.Queue, organize_files: int):
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
        :param organize_files: Perform alphabetical directory indexing of provided level.
            Expand output path by subfolders using the first n characters of repository,
            for example for "organize_files=2" file ababa is saved to /a/ab/ababa, bcoasa
            is saved to /b/bc/bcoasa, etc.
        :return: The child process' exit code.
        """
        if "log_level" in args:
            setup_logging(args.get("log_level"))
        if "grpc" in sys.modules:
            logging.getLogger(cls.__name__).error("grpc detected, fork() is unstable -> aborted")
            queue.put((url_or_path, 0))
            return 0
        pid = os.fork()
        if pid == 0:
            outfile = cls.prepare_filename(url_or_path, outdir, organize_files)
            worker = cls(**args)
            status = worker.process_repo(url_or_path, outfile)
            if multiprocessing.get_start_method() == "fork":
                sys.exit(status)
            os._exit(status)
        else:
            _, status = os.waitpid(pid, 0)
            queue.put((url_or_path, status))
            return status

    @classmethod
    def prepare_filename(cls, repo: str, output: str, organize_files: int=0):
        """
        Remove prefixes from the repo name, so later it can be used to create
        file for each repository + replace slashes ("/") with ampersands ("&").

        :param repo: name of repository
        :param output: output directory
        :param organize_files: Perform alphabetical directory indexing of provided level.
            Expand output path by subfolders using the first n characters of repository,
            for example for "organize_files=2" file ababa is saved to /a/ab/ababa, bcoasa
            is saved to /b/bc/bcoasa, etc.
        :return: converted repository name (removed "https://", etc.)
        """
        if os.path.exists(repo):
            repo_name = os.path.split(repo.rstrip("/\\"))[-1]
        else:
            repo_name = repo
            prefixes = ["https://", "http://", "git://", "ssh://"]
            for prefix in prefixes:
                if repo.startswith(prefix):
                    repo_name = repo_name[len(prefix):]
                    break
            sites = ["bitbucket.org/", "github.com/"]
            for site in sites:
                if repo_name.startswith(site):
                    repo_name = repo_name[len(site):]
                    break
            postfixes = ["\n", "/", "\\", ".git"]
            for postfix in postfixes:
                if repo_name.endswith(postfix):
                    repo_name = repo_name[:-len(postfix)]
        output = os.path.join(output, *list(accumulate(repo_name[:organize_files])))
        os.makedirs(output, exist_ok=True)
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
        overwrite_existing = self._args.get("overwrite_existing",
                                            self.WORKER_CLASS.DEFAULT_OVERWRITE_EXISTING)
        if os.path.exists(output):
            if overwrite_existing:
                self._log.warning("Model %s already exists, but will be overwrite. If you want to "
                                  "skip existing models use --disable-overwrite flag", output)
            else:
                self._log.warning("Model %s already exists, skipping.", output)
                return True
        try:
            repo2 = self.WORKER_CLASS(**self._args)
            result = repo2.convert_repository(url_or_path)
            for proto in ("https://", "http://", "git://", "ssh://"):
                if url_or_path.startswith(proto):
                    url_or_path = url_or_path.replace(proto, "")
            model = self.WORKER_CLASS.MODEL_CLASS()
            model.construct(**self.result_to_model_kwargs(result, url_or_path))
            if self._log.isEnabledFor(logging.DEBUG):
                self._log.debug("Save %s model...", url_or_path)
                start = time()
            model.save(output, deps=self.dependencies())
            if self._log.isEnabledFor(logging.DEBUG):
                self._log.debug("Save %s model is done. time: %f", url_or_path, time() - start)
            return True
        except subprocess.CalledProcessError as e:
            self._log.error("Failed to clone %s: %s", url_or_path, e)
            return False
        except ValueError as e:
            self._log.warning("Failed to construct model for %s: %s", url_or_path, e)
            return False
        except:
            self._log.exception(
                "Unhandled error in %s.process_repo() at %s." % (
                    type(self).__name__, url_or_path))
            return False

    def transform(self, repos, output, num_processes=None):
        """
        Converts repositories to models and saves them to the output directory.

        :param repos: "repos" is the list of repository URLs or paths or
                  files with repository URLS or paths.
        :param output: "output" is the output directory where to store the
                        results.
        :param num_processes: number of processes to use, if negative - use all
               CPUs.
        :return: None
        """
        time_start = time()
        self._args["log_level"] = self._log.level
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
        failures = 0
        with multiprocessing.Pool(processes=num_processes) as pool:
            pool.starmap_async(
                type(self).process_entry,
                zip(inputs, repeat(self._args), repeat(output), repeat(queue),
                    repeat(self._organize_files)))

            for _ in progress_bar(inputs, self._log, expected_size=len(inputs)):
                repo, ok = queue.get()
                if not ok:
                    failures += 1

        self._log.info("Finished, %d failed repos", failures)
        if self.WORKER_CLASS.PARSE_LOG_FILENAME and os.path.exists(
                self.WORKER_CLASS.PARSE_LOG_FILENAME):
            lines, time_spent = parse_parse_logs(self.WORKER_CLASS.PARSE_LOG_FILENAME)
            threads = self._args.get("threads", self.WORKER_CLASS.DEFAULT_THREAD_COUNT)
            self._log.debug("Average bblfsh  speed % 6.2f lines/sec." % (lines / time_spent))
            self._log.debug("Average ast2vec speed % 6.2f lines/sec. %d process, %d threads" %
                            (lines / (time() - time_start), self.num_processes, threads))
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


def parse_parse_logs(filepath):
    lines = 0
    time_spent = 0
    with open(filepath) as f:
        f.readline()
        for line in f:
            sl = line.split(",")
            lines += int(sl[1])
            time_spent += float(sl[2])
    return lines, time_spent


def lines_count(filepath, log=None):
    """
    Returns line count in file or -1 if error.

    :param filepath: path to file.
    :pasram log: For exception output.
    :return: lines count
    """
    try:
        with open(filepath, encoding="utf8") as f:
            file_content = f.read()
            return file_content.count("\n") + 0 if file_content[-1] == "\n" else 1
    except Exception as e:
        if log:
            log.exception("Can not get size for %s" % filepath)
        return float("nan")


def resolve_bblfsh_timeout(bblfsh_timeout):
    if bblfsh_timeout is not None:
        return bblfsh_timeout
    log = logging.getLogger("bblfsh")
    env_timeout = os.getenv("BBLFSH_TIMEOUT")
    if env_timeout:
        try:
            log.debug("Got bblfsh timeout from BBLFSH_TIMEOUT environment variable: %s sec",
                      env_timeout)
            return int(env_timeout)
        except ValueError as e:
            log.warning(("You provide wrong value: %s in BBLFSH_TIMEOUT environment variable. "
                        "Should be integer. Default %s sec will be used", env_timeout,
                         DEFAULT_BBLFSH_TIMEOUT))
    log.debug("You did not provide bblfsh timeout directly or in BBLFSH_TIMEOUT environment "
              "variable. Default %s sec will be used", DEFAULT_BBLFSH_TIMEOUT)
    return DEFAULT_BBLFSH_TIMEOUT


def resolve_bblfsh_endpoint(bblfsh_endpoint):
    if bblfsh_endpoint is not None:
        return bblfsh_endpoint
    log = logging.getLogger("bblfsh")
    env_endpoint = os.getenv("BBLFSH_ENDPOINT")
    if env_endpoint:
        log.debug("Got bblfsh endpoint from BBLFSH_ENDPOINT environment variable: %s",
                  env_endpoint)
        return env_endpoint
    log.debug("You did not provide the bblfsh endpoint directly or in BBLFSH_ENDPOINT "
              "environment variable. Default %s will be used", DEFAULT_BBLFSH_ENDPOINTS)
    for addr in DEFAULT_BBLFSH_ENDPOINTS:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        host, port = addr.split(":")
        log.debug("Probing %s", addr)
        result = sock.connect_ex((host, int(port)))
        if result == 0:
            log.info("Detected bblfsh server: %s", addr)
            return addr
    return DEFAULT_BBLFSH_ENDPOINTS[0]


def ensure_bblfsh_is_running_noexc(bblfsh_endpoint=None):
    """
    Launches the Babelfish server, if it is possible and needed.

    :param bblfsh_endpoint: bblfsh endpoint to check.
    :return: None
    """
    if resolve_bblfsh_endpoint(bblfsh_endpoint) != DEFAULT_BBLFSH_ENDPOINTS[0]:
        return
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

    :param args: :class:`argparse.Namespace` with "repository" and "output".
                 "repository" is a file system path or a URL. "output" is the path
                 to the file with the resulting model.
    :param payload_class: :class:`Transformer` inheritor to call.
    :return: None
    """
    ensure_bblfsh_is_running_noexc(args.bblfsh_endpoint)
    payload_args = _sanitize_kwargs(args, "repository")
    payload_class(**payload_args).process_repo(args.repository, args.output)


def repos2_entry(args, payload_class):
    """
    Invokes payload_class(\*\*args).transform() for every repository in parallel processes.

    :param args: :class:`argparse.Namespace` with "input" and "output".
                 "input" is the list of repository URLs or paths or files
                 with repository URLS or paths. "output" is the output
                 directory where to store the results.
    :param payload_class: :class:`Transformer` inheritor to call.
    :return: None
    """
    ensure_bblfsh_is_running_noexc(args.bblfsh_endpoint)
    payload_args = _sanitize_kwargs(args, "input")
    payload_class(**payload_args).transform(args.input, args.output)
