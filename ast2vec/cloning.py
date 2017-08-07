from functools import partial
import json
import logging
from multiprocessing.dummy import Pool
import os
import shutil
import subprocess


class LinguistFailedError(Exception):
    """
    Raised when we fail to classify the source files.
    """
    pass


class RepoCloner:
    """
    Clones repositories from provided urls / files with urls.
    Use enry to classify files and delete redundant files if needed.
    """
    def __init__(self, redownload, linguist=None, languages=None,
                 log_level=logging.INFO, num_threads=1):
        self._log = logging.getLogger("repo_cloner")
        self._log.setLevel(log_level)
        self._is_enry = False
        self._languages = languages
        self._linguist = None
        self._num_threads = num_threads
        self._redownload = redownload
        if linguist or languages:
            self.find_linguist(linguist)

    def clone_repo(self, git_url, ignore, target_dir):
        """
        Clones repository into a separate directory inside of the target one.

        :param git_url: Url of Git repository.
        :param ignore: Flag for ignoring Git failures.
        :param target_dir: Target directory. New directory will be created inside of target_dir.
        :return: Path to downloaded Git repository.
        """
        git_url = self._prepare_repo_url(git_url)

        try:
            repo_dir = self._prepare_repo_dir(git_url, target_dir)
        except FileExistsError as e:
            self._log.warning("%s already cloned, skipping.", git_url)
            return

        env = os.environ.copy()
        env["GIT_TERMINAL_PROMPT"] = "0"  # force Git not to ask anything
        self._log.info("Cloning from %s...", git_url)

        try:
            subprocess.check_output(["git", "clone", "--depth=1", git_url, repo_dir],
                                    env=env, stdin=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            shutil.rmtree(repo_dir, ignore_errors=True)
            self._log.error("Git failed to clone repo. git stderr:\n\t" +
                            "\n\t".join(e.output.decode("utf8").split("\n")))
            if not ignore:
                raise e from None
            else:
                return
        except Exception as e:
            shutil.rmtree(repo_dir, ignore_errors=True)
            self._log.error("Unknown error in RepoCloner. Git failed to clone repo.")
            raise e from None

        self._log.info("Finished cloning %s", git_url)
        return repo_dir

    def classify_repo(self, repo_dir: str) -> dict:
        """
        Classify files in a repository using provided linguist.

        :param repo_dir: Path to repository directory.
        :return: linguist output loaded from JSON.
        """
        if not self._linguist:
            raise LinguistFailedError("Linguist is not set - cannot classify the files")

        self._log.info("Classifying the files...")
        repo_dir = os.path.abspath(repo_dir)
        cmdline = [self._linguist]
        if self._is_enry:
            cmdline += ["-json", repo_dir]
        else:
            cmdline += [repo_dir, "--json"]
        try:
            bjson = subprocess.check_output(cmdline)
        except subprocess.CalledProcessError:
            self._log.error("Couldn't classify files in %s", repo_dir)
            raise LinguistFailedError() from None
        classified = json.loads(bjson.decode("utf8"))
        self._log.info("Result: %s", {k: len(v) for k, v in classified.items()})
        return classified

    def cleanup_repo(self, classified: dict, repo_dir: str) -> None:
        """
        Delete files not classified by linguist (if some languages were specified then only them
        are preserved).

        :param classified: enry output loaded from JSON.
        :param repo_dir: Path to repository directory.
        :return:
        """
        if self._languages:
            self._log.info("Removing files with languages not in (%s)",
                           ",".join(self._languages))
            languages = self._languages
        else:
            self._log.info("Removing files not classified by enry.")
            languages = classified.keys()

        allowed_files = set(str.encode(os.path.join(repo_dir, fname)) for lang in languages
                            if lang in classified for fname in classified[lang])

        for root, dirnames, filenames in os.walk(str.encode(repo_dir), topdown=False):
            for filename in filenames:
                full_filename = os.path.join(root, filename)
                if full_filename not in allowed_files:
                    os.remove(full_filename)
            for dirname in dirnames:
                full_dirname = os.path.join(root, dirname)
                if os.path.islink(full_dirname):
                    os.unlink(full_dirname)
                elif not os.listdir(full_dirname):
                    os.rmdir(full_dirname)

    def clone_repos(self, inputs, output, ignore):
        with Pool(self._num_threads) as pool:
            pool.map(partial(self.process_repo, ignore=ignore, target_dir=output),
                     self.generate_repo_urls(inputs))

    def process_repo(self, git_url: str, ignore: bool, target_dir: str) -> None:
        repo_dir = self.clone_repo(git_url, ignore, target_dir)
        if repo_dir is None or self._linguist is None:
            return
        classified = self.classify_repo(repo_dir)
        if classified:
            self.cleanup_repo(classified, repo_dir)

    @staticmethod
    def generate_repo_urls(inputs):
        """
        Parse provided inputs.

        :param inputs: List of files and/or Git urls.
        :return: Generator of git urls.
        """
        for item in inputs:
            if os.path.isfile(item):
                with open(item, encoding="utf8") as f:
                    for line in f:
                        yield line.rstrip()
            else:
                yield item

    def find_linguist(self, linguist):
        if linguist is None:
            linguist = shutil.which("enry", path=os.getcwd() + ":" + os.getenv("PATH", os.defpath))
        full_path = shutil.which(linguist)
        if not full_path:
            raise FileNotFoundError("%s was not found. Install it: python3 -m ast2vec enry" %
                                    linguist)
        self._linguist = linguist
        with open(full_path, "rb") as fin:
            # Check if we're using https://github.com/github/linguist
            self._is_enry = fin.read(15) != b"#!/usr/bin/ruby"

    def _prepare_repo_dir(self, git_url: str, target_dir: str) -> str:
        """
        Prepare directory for saving Git repository, i.e. create / cleanup if necessary.

        :param git_url: Url of Git repository.
        :param target_dir: Parent directory for Git repository.
        :return: Path to prepared directory.
        """
        git_ending = ".git"
        repo_name = "@".join(git_url.rsplit("/", maxsplit=2)[-2:])
        if repo_name.endswith(git_ending):
            repo_name = repo_name[:-len(git_ending)]
        repo_dir = os.path.join(target_dir, repo_name)

        if os.path.exists(repo_dir) and self._redownload:
            self._log.info("%s already downloaded to %s, will download it again",
                           git_url, repo_dir)
            shutil.rmtree(repo_dir)

        os.makedirs(repo_dir)
        return repo_dir

    @staticmethod
    def _prepare_repo_url(git_url: str) -> str:
        """
        Prepare name of repository for operations with git.
        Remove '\n', '/' and '\' in the end of string.
        Add 'https://' in the beginning if necessary.

        :param reponame: Raw Git url of repository.
        :return: Clean Git url.
        """
        bad_endings = "\n\r\\/"
        git_url = git_url.rstrip(bad_endings)
        if not git_url.startswith("git://") and not git_url.startswith("https://") and \
                not git_url.startswith("http://"):
            git_url = "https://" + git_url
        return git_url


def clone_repositories(args):
    """
    Invokes RepoCloner(\*\*args).clone_repos() on the specified input.

    :param args: :class:`argparse.Namespace` with "input", "output" and "ignore". "input" is a \
                 list of files and/or Git urls. "output" is the path to directory for storing \
                 all repositories. "ignore" is a flag for specifying to ignore Git clone problems.
    :return: None
    """
    clone_args = _sanitize_kwargs(args)
    RepoCloner(**clone_args).clone_repos(args.input, args.output, args.ignore)


def _sanitize_kwargs(args):
    clone_args = getattr(args, "__dict__", args).copy()
    blacklist = ("command", "ignore", "input", "handler", "output")
    for arg in blacklist:
        clone_args.pop(arg, None)
    clone_args["num_threads"] = clone_args.pop("threads")
    return clone_args
