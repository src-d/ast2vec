import argparse
import logging
import os
import sys

from ast2vec.vw_dataset import bow2vw_entry
from modelforge.logs import setup_logging

from ast2vec.cloning import clone_repositories
from ast2vec.dump import dump_model
from ast2vec.enry import install_enry
from ast2vec.id_embedding import preprocess, run_swivel, postprocess, swivel
from ast2vec.repo2.base import Repo2Base
from ast2vec.repo2.coocc import repo2coocc_entry, repos2coocc_entry
from ast2vec.repo2.nbow import repo2nbow_entry, repos2nbow_entry
from ast2vec.repo2.uast import repo2uast_entry, repos2uast_entry
from ast2vec.model2.join_bow import joinbow_entry
from ast2vec.model2.prox import prox_entry, MATRIX_TYPES
from ast2vec.model2.proxbase import EDGE_TYPES


def main():
    """
    Creates all the argparse-rs and invokes the function from set_defaults().

    :return: The result of the function from set_defaults().
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-level", default="INFO",
                        choices=logging._nameToLevel,
                        help="Logging verbosity.")
    subparsers = parser.add_subparsers(help="Commands", dest="command")

    clone_parser = subparsers.add_parser(
        "clone", help="Clone multiple repositories. By default saves all files, including "
        "`.git`. Use --linguist and --languages options to narrow files down.")
    clone_parser.set_defaults(handler=clone_repositories)
    clone_parser.add_argument(
        "input", nargs="+", help="List of repositories and/or files with list of repositories.")
    clone_parser.add_argument(
        "--ignore", action="store_true",
        help="Ignore failed to download repositories. An error message is logged as usual.")
    clone_parser.add_argument(
        "--linguist", help="Path to src-d/enry executable. If specified will save only files "
        "classified by enry.")
    clone_parser.add_argument(
        "--languages", nargs="*", default=["Python", "Java"], help="Files which are classified "
        "as not written in these languages are discarded.")
    clone_parser.add_argument(
        "-o", "--output", required=True, help="Output directory.")
    clone_parser.add_argument(
        "--redownload", action="store_true", help="Redownload existing repositories.")
    clone_parser.add_argument(
        "-t", "--threads", type=int, required=True, help="Number of downloading threads.")

    repo2nbow_parser = subparsers.add_parser(
        "repo2nbow", help="Produce the nBOW from a Git repository.")
    repo2nbow_parser.set_defaults(handler=repo2nbow_entry)
    repo2nbow_parser.add_argument(
        "repository", help="URL or path to a Git repository.")
    repo2nbow_parser.add_argument(
        "--id2vec", help="URL or path to the identifier embeddings.")
    repo2nbow_parser.add_argument(
        "--df", dest="docfreq", help="URL or path to the document frequencies.")
    repo2nbow_parser.add_argument(
        "--linguist", help="Path to src-d/enry executable.")
    repo2nbow_parser.add_argument(
        "--bblfsh", help="Babelfish server's endpoint, e.g. 0.0.0.0:9432.",
        dest="bblfsh_endpoint")
    repo2nbow_parser.add_argument(
        "--timeout", type=int, default=Repo2Base.DEFAULT_BBLFSH_TIMEOUT,
        help="Babelfish timeout - longer requests are dropped.")
    repo2nbow_parser.add_argument(
        "-o", "--output", required=True,
        help="Output path where the .asdf will be stored.")
    repo2nbow_parser.add_argument("--gcs", default=None, dest="gcs_bucket",
                                  help="GCS bucket to use.")

    repos2nbow_parser = subparsers.add_parser(
        "repos2nbow", help="Produce the nBOWs from a list of Git "
                           "repositories.")
    repos2nbow_parser.set_defaults(handler=repos2nbow_entry)
    repos2nbow_parser.add_argument(
        "input", nargs="+", help="List of repositories and/or files with list of repositories.")
    repos2nbow_parser.add_argument(
        "--id2vec", help="URL or path to the identifier embeddings.")
    repos2nbow_parser.add_argument(
        "--df", dest="docfreq", help="URL or path to the document frequencies.")
    repos2nbow_parser.add_argument(
        "--linguist", help="Path to src-d/enry executable.")
    repos2nbow_parser.add_argument(
        "-o", "--output", required=True,
        help="Output directory where .asdf results will be stored.")
    repos2nbow_parser.add_argument(
        "--bblfsh", help="Babelfish server's endpoint, e.g. 0.0.0.0:9432.",
        dest="bblfsh_endpoint")
    repos2nbow_parser.add_argument(
        "--timeout", type=int, default=Repo2Base.DEFAULT_BBLFSH_TIMEOUT,
        help="Babelfish timeout - longer requests are dropped.")
    repos2nbow_parser.add_argument("--gcs", default=None, dest="gcs_bucket",
                                   help="GCS bucket to use.")
    repos2nbow_parser.add_argument(
        "-p", "--processes", type=int, default=2, dest="num_processes",
        help="Number of parallel processes to run. Since every process "
             "spawns the number of threads equal to the number of CPU cores "
             "it is better to set this to 1 or 2.")

    repo2coocc_parser = subparsers.add_parser(
        "repo2coocc", help="Produce the co-occurrence matrix from a Git "
                           "repository.")
    repo2coocc_parser.set_defaults(handler=repo2coocc_entry)
    repo2coocc_parser.add_argument(
        "repository", help="URL or path to a Git repository.")
    repo2coocc_parser.add_argument(
        "--linguist", help="Path to src-d/enry executable.")
    repo2coocc_parser.add_argument(
        "-o", "--output", required=True,
        help="Output path where .asdf result will be stored.")
    repo2coocc_parser.add_argument(
        "--bblfsh", help="Babelfish server's endpoint, e.g. 0.0.0.0:9432.",
        dest="bblfsh_endpoint")
    repo2coocc_parser.add_argument(
        "--timeout", type=int, default=Repo2Base.DEFAULT_BBLFSH_TIMEOUT,
        help="Babelfish timeout - longer requests are dropped.")

    repos2coocc_parser = subparsers.add_parser(
        "repos2coocc", help="Produce the co-occurrence matrix from a list of "
                            "Git repositories.")
    repos2coocc_parser.set_defaults(handler=repos2coocc_entry)
    repos2coocc_parser.add_argument(
        "input", nargs="+", help="List of repositories and/or files with list of repositories.")
    repos2coocc_parser.add_argument(
        "--linguist", help="Path to src-d/enry executable.")
    repos2coocc_parser.add_argument(
        "-o", "--output", required=True,
        help="Output directory where .asdf results will be stored.")
    repos2coocc_parser.add_argument(
        "--bblfsh", help="Babelfish server's endpoint, e.g. 0.0.0.0:9432.",
        dest="bblfsh_endpoint")
    repos2coocc_parser.add_argument(
        "--timeout", type=int, default=Repo2Base.DEFAULT_BBLFSH_TIMEOUT,
        help="Babelfish timeout - longer requests are dropped.")
    repos2coocc_parser.add_argument(
        "-p", "--processes", type=int, default=2, dest="num_processes",
        help="Number of parallel processes to run. Since every process "
             "spawns the number of threads equal to the number of CPU cores "
             "it is better to set this to 1 or 2.")

    repo2uast_parser = subparsers.add_parser(
        "repo2uast", help="Extract UASTs from a Git repository.")
    repo2uast_parser.set_defaults(handler=repo2uast_entry)
    repo2uast_parser.add_argument("repository", help="URL or path to a Git repository.")
    repo2uast_parser.add_argument("--linguist", help="Path to src-d/enry executable.")
    repo2uast_parser.add_argument(
        "-o", "--output", required=True,
        help="Output path where the .asdf will be stored.")
    repo2uast_parser.add_argument(
        "--bblfsh", help="Babelfish server's endpoint, e.g. 0.0.0.0:9432.",
        dest="bblfsh_endpoint")
    repo2uast_parser.add_argument(
        "--timeout", type=int, default=Repo2Base.DEFAULT_BBLFSH_TIMEOUT,
        help="Babelfish timeout - longer requests are dropped.")

    repos2uast_parser = subparsers.add_parser(
        "repos2uast", help="Extract UASTs from a list of Git repositories.")
    repos2uast_parser.set_defaults(handler=repos2uast_entry)
    repos2uast_parser.add_argument(
        "input", nargs="+", help="List of repositories and/or files with list of repositories.")
    repos2uast_parser.add_argument("--linguist", help="Path to src-d/enry executable.")
    repos2uast_parser.add_argument(
        "-o", "--output", required=True,
        help="Output directory where .asdf results will be stored.")
    repos2uast_parser.add_argument(
        "--bblfsh", help="Babelfish server's endpoint, e.g. 0.0.0.0:9432.",
        dest="bblfsh_endpoint")
    repos2uast_parser.add_argument(
        "--timeout", type=int, default=Repo2Base.DEFAULT_BBLFSH_TIMEOUT,
        help="Babelfish timeout - longer requests are dropped.")

    joinbow_parser = subparsers.add_parser(
        "join_bow", help="Combine several nBOW files into the single one.")
    joinbow_parser.set_defaults(handler=joinbow_entry)
    joinbow_parser.add_argument(
        "input", help="Directory to scan recursively.")
    joinbow_parser.add_argument(
        "output", help="Where to write the merged nBOW.")
    group = joinbow_parser.add_argument_group("type")
    group_ex = group.add_mutually_exclusive_group(required=True)
    group_ex.add_argument("--bow", action="store_true", help="The models are BOW.")
    group_ex.add_argument("--nbow", action="store_true", help="The models are NBOW.")
    joinbow_parser.add_argument(
        "-p", "--processes", type=int, default=0,
        help="Number of processes to use. 0 means CPU count,")
    joinbow_parser.add_argument(
        "--tmpdir", default=None, help="Temporary directory for intermediate files.")
    joinbow_parser.add_argument(
        "--filter", default="**/*.asdf", help="File name glob selector.")

    uast2prox_parser = subparsers.add_parser(
        "uast2prox", help="Convert UASTs to proximity matrix.")
    uast2prox_parser.set_defaults(handler=prox_entry)
    uast2prox_parser.add_argument("input", help="Directory to scan recursively for UASTs.")
    uast2prox_parser.add_argument("output", help="Where to write the resulting proximity matrix.")
    uast2prox_parser.add_argument(
        "-m", "--matrix-type", required=True, choices=MATRIX_TYPES.keys(),
        help="Type of proximity matrix.")
    uast2prox_parser.add_argument(
        "-p", "--processes", type=int, default=0,
        help="Number of processes to use. 0 means CPU count,")
    uast2prox_parser.add_argument(
        "--edges", nargs="+", default=EDGE_TYPES, choices=EDGE_TYPES,
        help="If not specified, then node-to-node adjacency is assumed. Suppose we have two "
        "connected nodes A and B:\n"
        "r - connect node roles with each other.\n"
        "t - connect node tokens with each other.\n"
        "rt - connect node tokens with node roles.\n"
        "R - connect node A roles with node B roles.\n"
        "T - connect node A tokens with node B tokens.\n"
        "RT - connect node A roles(tokens) with node B tokens(roles).")
    uast2prox_parser.add_argument(
        "--filter", default="**/*.asdf", help="File name glob selector.")

    preproc_parser = subparsers.add_parser(
        "id2vec_preproc", help="Convert co-occurrence CSR matrices to Swivel dataset.")
    preproc_parser.set_defaults(handler=preprocess)
    preproc_parser.add_argument(
        "-o", "--output", required=True, help="The output directory.")
    preproc_parser.add_argument(
        "-v", "--vocabulary-size", default=1 << 17, type=int,
        help="The final vocabulary size. Only the most frequent words will be"
             "left.")
    preproc_parser.add_argument("-s", "--shard-size", default=4096, type=int,
                                help="The shard (submatrix) size.")
    preproc_parser.add_argument(
        "--df", default=None,
        help="Path to the calculated document frequencies in asdf format "
             "(DF in TF-IDF).")
    preproc_parser.add_argument(
        "input", nargs="+",
        help="Pickled scipy.sparse matrices. If it is a directory, all files "
             "inside are read.")

    train_parser = subparsers.add_parser(
        "id2vec_train", help="Train identifier embeddings.")
    train_parser.set_defaults(handler=run_swivel)
    del train_parser._action_groups[train_parser._action_groups.index(
        train_parser._optionals)]
    train_parser._optionals = swivel.flags._global_parser._optionals
    train_parser._action_groups.append(train_parser._optionals)
    train_parser._actions = swivel.flags._global_parser._actions
    train_parser._option_string_actions = \
        swivel.flags._global_parser._option_string_actions

    postproc_parser = subparsers.add_parser(
        "id2vec_postproc",
        help="Combine row and column embeddings together and write them to an .asdf.")
    postproc_parser.set_defaults(handler=postprocess)
    postproc_parser.add_argument("swivel_output_directory")
    postproc_parser.add_argument("result")

    bow2vw_parser = subparsers.add_parser(
        "bow2vw", help="Convert a bag-of-words model to the dataset in Vowpal Wabbit format.")
    bow2vw_parser.set_defaults(handler=bow2vw_entry)
    group = bow2vw_parser.add_argument_group("model")
    group_ex = group.add_mutually_exclusive_group(required=True)
    group_ex.add_argument(
        "--bow", help="URL or path to a bag-of-words model. Mutually exclusive with --nbow.")
    group_ex.add_argument(
        "--nbow", help="URL or path to an nBOW model. Mutually exclusive with --bow.")
    bow2vw_parser.add_argument(
        "--id2vec", help="URL or path to the identifier embeddings. Used if --nbow")
    bow2vw_parser.add_argument(
        "-o", "--output", required=True, help="Path to the output file.")

    enry_parser = subparsers.add_parser(
        "enry", help="Install src-d/enry to the current working directory.")
    enry_parser.set_defaults(handler=install_enry)
    enry_parser.add_argument(
        "--tempdir",
        help="Store intermediate files in this directory instead of /tmp.")
    enry_parser.add_argument("--output", default=os.getcwd(),
                             help="Output directory.")

    dump_parser = subparsers.add_parser(
        "dump", help="Dump a model to stdout.")
    dump_parser.set_defaults(handler=dump_model)
    dump_parser.add_argument(
        "input", help="Path to the model file, URL or UUID.")
    dump_parser.add_argument(
        "-d", "--dependency", nargs="+",
        help="Paths to the models which were used to generate the dumped model in "
             "the order they appear in the metadata.")
    dump_parser.add_argument("--gcs", default=None, help="GCS bucket to use.")

    args = parser.parse_args()
    args.log_level = logging._nameToLevel[args.log_level]
    setup_logging(args.log_level)
    try:
        handler = args.handler
    except AttributeError:
        def print_usage(_):
            parser.print_usage()

        handler = print_usage
    return handler(args)

if __name__ == "__main__":
    sys.exit(main())
