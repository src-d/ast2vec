import argparse
import logging
import multiprocessing
import os
import sys

from ast2vec.id2vec import projector_entry
from ast2vec.topics import bigartm2asdf_entry
from modelforge.logs import setup_logging

from ast2vec.bigartm import install_bigartm
from ast2vec.cloning import clone_repositories
from ast2vec.dump import dump_model
from ast2vec.enry import install_enry
from ast2vec.id_embedding import preprocess as preprocess_id2vec, run_swivel, \
    postprocess as postprocess_id2vec, swivel
from ast2vec.vw_dataset import bow2vw_entry
from ast2vec.repo2.base import Repo2Base, RepoTransformer, \
    DEFAULT_BBLFSH_TIMEOUT, DEFAULT_BBLFSH_ENDPOINTS
from ast2vec.repo2.coocc import repo2coocc_entry, repos2coocc_entry
from ast2vec.repo2.nbow import repo2nbow_entry, repos2nbow_entry
from ast2vec.repo2.uast import repo2uast_entry, repos2uast_entry
from ast2vec.repo2.source import repo2source_entry, repos2source_entry
from ast2vec.model2.join_bow import joinbow_entry
from ast2vec.model2.prox import prox_entry, MATRIX_TYPES
from ast2vec.model2.proxbase import EDGE_TYPES
from ast2vec.model2.uast2bow import uast2bow_entry
from ast2vec.model2.uast2df import uast2df_entry


class ArgumentDefaultsHelpFormatterNoNone(argparse.ArgumentDefaultsHelpFormatter):
    """
    Pretty formatter of help message for arguments.
    It adds default value to the end if it is not None.
    """
    def _get_help_string(self, action):
        if action.default is None:
            return action.help
        return super()._get_help_string(action)


def one_arg_parser(*args, **kwargs) -> argparse.ArgumentParser:
    """
    Create parser for one argument with passed arguments.
    It is helper function to avoid argument duplication in subcommands.

    :return: Parser for one argument.
    """
    arg_parser = argparse.ArgumentParser(add_help=False)
    arg_parser.add_argument(*args, **kwargs)
    return arg_parser


def get_parser() -> argparse.ArgumentParser:
    """
    Create main parser.

    :return: Parser
    """
    parser = argparse.ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatterNoNone)
    parser.add_argument("--log-level", default="INFO",
                        choices=logging._nameToLevel,
                        help="Logging verbosity.")

    # Create all common arguments

    repository_arg = one_arg_parser(
        "repository", help="URL or path to a Git repository.")
    repos2input_arg = one_arg_parser(
        "input", nargs="+", help="List of repositories and/or files with list of repositories.")
    model2input_arg = one_arg_parser(
        "input", help="Directory to scan recursively for asdf files.")

    output_dir_arg_default = one_arg_parser(
        "-o", "--output", required=True, help="Output directory.")
    output_dir_arg_asdf = one_arg_parser(
        "-o", "--output", required=True, help="Output path where the .asdf will be stored.")

    bblfsh_args = argparse.ArgumentParser(add_help=False)
    bblfsh_args.add_argument(
        "--bblfsh", dest="bblfsh_endpoint",
        help="Babelfish server's endpoint, e.g. 0.0.0.0:9432. "
             "You can specify it directly or with BBLFSH_ENDPOINT environment variable. Otherwise "
             "default will be used (default: %s)" % DEFAULT_BBLFSH_ENDPOINTS)
    bblfsh_args.add_argument(
        "--timeout", type=int,
        help="Babelfish timeout - longer requests are dropped. "
             "You can specify it directly or with BBLFSH_TIMEOUT environment variable. Otherwise "
             "default will be used (default: %d sec)" % DEFAULT_BBLFSH_TIMEOUT)

    process_arg = one_arg_parser(
        "-p", "--processes", type=int, default=0,
        help="Number of processes to use. 0 means CPU count.")
    process_1_2_arg = one_arg_parser(
        "-p", "--processes", type=int, default=2, dest="num_processes",
        help="Number of parallel processes to run. Since every process "
             "spawns the number of threads equal to the number of CPU cores "
             "it is better to set this to 1 or 2.")
    threads_arg = one_arg_parser(
        "-t", "--threads", type=int, default=multiprocessing.cpu_count(),
        help="Number of threads in the UASTs extraction process.")

    organize_files_arg = one_arg_parser(
        "--organize-files", type=int, default=RepoTransformer.DEFAULT_ORGANIZE_FILES,
        help="Perform alphabetical directory indexing of provided level. Expand output path by "
             "subfolders using the first n characters of repository, for example for "
             "\"--organize-files 2\" file ababa is saved to /a/ab/ababa, bcoasa is saved to "
             "/b/bc/bcoasa, etc.")

    disable_overwrite_arg = one_arg_parser(
        "--disable-overwrite", action="store_false", default=Repo2Base.DEFAULT_OVERWRITE_EXISTING,
        dest="overwrite_existing",
        help="Specify if you want to disable overiting of existing models")

    bblfsh_raise_arg = one_arg_parser(
        "--bblfsh-raise", action="store_true", default=Repo2Base.DEFAULT_BBLFSH_RAISE_ERRORS,
        dest="bblfsh_raise_errors",
        help="Specify if you want to raise errors upon receiving errors from bblfsh server")

    linguist_arg = one_arg_parser(
        "--linguist", help="Path to src-d/enry executable.")

    gcs_arg = one_arg_parser("--gcs", default=None, dest="gcs_bucket",
                             help="GCS bucket to use.")

    tmpdir_arg = one_arg_parser(
        "--tmpdir", help="Store intermediate files in this directory instead of /tmp.")

    filter_arg = one_arg_parser(
        "--filter", default="**/*.asdf", help="File name glob selector.")

    id2vec_arg = one_arg_parser(
        "--id2vec", help="URL or path to the identifier embeddings.")
    df_arg = one_arg_parser(
        "-d", "--df", dest="docfreq", help="URL or path to the document frequencies.")
    prune_arg = one_arg_parser(
        "--prune-df", default=20,
        help="Minimum document frequency to leave an identifier.")
    outputdir_arg = one_arg_parser("--output", default=os.getcwd(), help="Output directory.")

    # Create and construct subparsers

    subparsers = parser.add_subparsers(help="Commands", dest="command")

    clone_parser = subparsers.add_parser(
        "clone", help="Clone multiple repositories. By default saves all files, including "
        "`.git`. Use --linguist and --languages options to narrow files down.",
        formatter_class=ArgumentDefaultsHelpFormatterNoNone,
        parents=[repos2input_arg, output_dir_arg_default])
    clone_parser.set_defaults(handler=clone_repositories)
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
        "--redownload", action="store_true", help="Redownload existing repositories.")
    clone_parser.add_argument(
        "-t", "--threads", type=int, required=True, help="Number of downloading threads.")

    repo2nbow_parser = subparsers.add_parser(
        "repo2nbow", help="Produce the nBOW from a Git repository.",
        formatter_class=ArgumentDefaultsHelpFormatterNoNone,
        parents=[repository_arg, id2vec_arg, df_arg, linguist_arg, bblfsh_args,
                 output_dir_arg_asdf, gcs_arg, threads_arg, disable_overwrite_arg,
                 prune_arg])
    repo2nbow_parser.set_defaults(handler=repo2nbow_entry)

    repos2nbow_parser = subparsers.add_parser(
        "repos2nbow", help="Produce the nBOWs from a list of Git repositories.",
        formatter_class=ArgumentDefaultsHelpFormatterNoNone,
        parents=[repos2input_arg, id2vec_arg, df_arg, linguist_arg, output_dir_arg_asdf,
                 bblfsh_args, gcs_arg, process_1_2_arg, threads_arg, organize_files_arg,
                 disable_overwrite_arg, repos2input_arg, prune_arg])
    repos2nbow_parser.set_defaults(handler=repos2nbow_entry)

    repo2coocc_parser = subparsers.add_parser(
        "repo2coocc", help="Produce the co-occurrence matrix from a Git repository.",
        formatter_class=ArgumentDefaultsHelpFormatterNoNone,
        parents=[repository_arg, linguist_arg, output_dir_arg_asdf, bblfsh_args,
                 threads_arg, disable_overwrite_arg])
    repo2coocc_parser.set_defaults(handler=repo2coocc_entry)

    repos2coocc_parser = subparsers.add_parser(
        "repos2coocc", help="Produce the co-occurrence matrix from a list of Git repositories.",
        formatter_class=ArgumentDefaultsHelpFormatterNoNone,
        parents=[repos2input_arg, linguist_arg, output_dir_arg_asdf, bblfsh_args, process_1_2_arg,
                 threads_arg, organize_files_arg, disable_overwrite_arg])
    repos2coocc_parser.set_defaults(handler=repos2coocc_entry)

    repo2uast_parser = subparsers.add_parser(
        "repo2uast", help="Extract UASTs from a Git repository.",
        formatter_class=ArgumentDefaultsHelpFormatterNoNone,
        parents=[repository_arg, linguist_arg, output_dir_arg_asdf, bblfsh_args, process_1_2_arg,
                 threads_arg, organize_files_arg, disable_overwrite_arg, bblfsh_raise_arg])
    repo2uast_parser.set_defaults(handler=repo2uast_entry)

    repos2uast_parser = subparsers.add_parser(
        "repos2uast", help="Extract UASTs from a list of Git repositories.",
        formatter_class=ArgumentDefaultsHelpFormatterNoNone,
        parents=[repos2input_arg, linguist_arg, output_dir_arg_asdf, bblfsh_args, process_1_2_arg,
                 threads_arg, organize_files_arg, disable_overwrite_arg, bblfsh_raise_arg])
    repos2uast_parser.set_defaults(handler=repos2uast_entry)

    repo2source_parser = subparsers.add_parser(
        "repo2source", help="Extract source model from a Git repository.",
        formatter_class=ArgumentDefaultsHelpFormatterNoNone,
        parents=[repository_arg, linguist_arg, output_dir_arg_asdf, bblfsh_args,
                 process_1_2_arg, threads_arg, organize_files_arg, disable_overwrite_arg])
    repo2source_parser.set_defaults(handler=repo2source_entry)

    repos2source_parser = subparsers.add_parser(
        "repos2source", help="Extract source model from a list of Git repositories.",
        formatter_class=ArgumentDefaultsHelpFormatterNoNone,
        parents=[repos2input_arg, linguist_arg, output_dir_arg_asdf, bblfsh_args,
                 process_1_2_arg, threads_arg, organize_files_arg, disable_overwrite_arg])
    repos2source_parser.set_defaults(handler=repos2source_entry)

    joinbow_parser = subparsers.add_parser(
        "join-bow", help="Combine several nBOW files into the single one.",
        formatter_class=ArgumentDefaultsHelpFormatterNoNone,
        parents=[model2input_arg, process_arg, tmpdir_arg, filter_arg])
    joinbow_parser.set_defaults(handler=joinbow_entry)
    joinbow_parser.add_argument("output", help="Where to write the merged nBOW.")
    group = joinbow_parser.add_argument_group("type")
    group_ex = group.add_mutually_exclusive_group(required=True)
    group_ex.add_argument("--bow", action="store_true", help="The models are BOW.")
    group_ex.add_argument("--nbow", action="store_true", help="The models are NBOW.")

    uast2df_parser = subparsers.add_parser(
        "uast2df", help="Calculate identifier document frequencies from extracted uasts.",
        formatter_class=ArgumentDefaultsHelpFormatterNoNone,
        parents=[model2input_arg, filter_arg, tmpdir_arg, process_arg])
    uast2df_parser.set_defaults(handler=uast2df_entry)
    uast2df_parser.add_argument("output", help="Where to write document frequencies.")

    uast2prox_parser = subparsers.add_parser(
        "uast2prox", help="Convert UASTs to proximity matrix.",
        formatter_class=ArgumentDefaultsHelpFormatterNoNone,
        parents=[model2input_arg, process_arg, filter_arg, disable_overwrite_arg])
    uast2prox_parser.set_defaults(handler=prox_entry)
    uast2prox_parser.add_argument("output", help="Where to write the resulting proximity matrix.")
    uast2prox_parser.add_argument(
        "-m", "--matrix-type", required=True, choices=MATRIX_TYPES.keys(),
        help="Type of proximity matrix.")
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

    uast2bow_parser = subparsers.add_parser(
        "uast2bow", help="Calculate bag of words from extracted uasts.",
        formatter_class=ArgumentDefaultsHelpFormatterNoNone,
        parents=[model2input_arg, filter_arg, process_arg, df_arg, disable_overwrite_arg,
                 prune_arg])
    uast2bow_parser.set_defaults(handler=uast2bow_entry)
    uast2bow_parser.add_argument(
        "-v", "--vocabulary-size", required=True, type=int,
        help="Vocabulary size: the tokens with the highest document frequencies will be picked.")
    uast2bow_parser.add_argument(
        "output", help="Where to write the merged nBOW.")

    preproc_parser = subparsers.add_parser(
        "id2vec_preproc", help="Convert co-occurrence CSR matrices to Swivel dataset.",
        formatter_class=ArgumentDefaultsHelpFormatterNoNone,
        parents=[output_dir_arg_default])
    preproc_parser.set_defaults(handler=preprocess_id2vec)
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
        help="Cooccurrence model produced by repo(s)2coocc. If it is a directory, all files "
             "inside are read.")

    train_parser = subparsers.add_parser(
        "id2vec_train", help="Train identifier embeddings.",
        formatter_class=ArgumentDefaultsHelpFormatterNoNone)
    train_parser.set_defaults(handler=run_swivel)
    del train_parser._action_groups[train_parser._action_groups.index(
        train_parser._optionals)]
    train_parser._optionals = swivel.flags._global_parser._optionals
    train_parser._action_groups.append(train_parser._optionals)
    train_parser._actions = swivel.flags._global_parser._actions
    train_parser._option_string_actions = \
        swivel.flags._global_parser._option_string_actions

    id2vec_postproc_parser = subparsers.add_parser(
        "id2vec_postproc",
        help="Combine row and column embeddings together and write them to an .asdf.",
        formatter_class=ArgumentDefaultsHelpFormatterNoNone)
    id2vec_postproc_parser.set_defaults(handler=postprocess_id2vec)
    id2vec_postproc_parser.add_argument("swivel_output_directory")
    id2vec_postproc_parser.add_argument("result")

    id2vec_projector_parser = subparsers.add_parser(
        "id2vec_projector", help="Present id2vec model in Tensorflow Projector.",
        formatter_class=ArgumentDefaultsHelpFormatterNoNone)
    id2vec_projector_parser.set_defaults(handler=projector_entry)
    id2vec_projector_parser.add_argument("-i", "--input", required=True,
                                         help="id2vec model to present.")
    id2vec_projector_parser.add_argument("-o", "--output", required=True,
                                         help="Projector output directory.")
    id2vec_projector_parser.add_argument("--df", help="docfreq model to pick the most significant "
                                                      "identifiers.")
    id2vec_projector_parser.add_argument("--no-browser", action="store_true",
                                         help="Do not open the browser.")

    bow2vw_parser = subparsers.add_parser(
        "bow2vw", help="Convert a bag-of-words model to the dataset in Vowpal Wabbit format.",
        formatter_class=ArgumentDefaultsHelpFormatterNoNone)
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

    bigartm_postproc_parser = subparsers.add_parser(
        "bigartm2asdf", help="Convert a readable BigARTM model to Modelforge format.",
        formatter_class=ArgumentDefaultsHelpFormatterNoNone)
    bigartm_postproc_parser.set_defaults(handler=bigartm2asdf_entry)
    bigartm_postproc_parser.add_argument("input")
    bigartm_postproc_parser.add_argument("output")

    enry_parser = subparsers.add_parser(
        "enry", help="Install src-d/enry to the current working directory.",
        formatter_class=ArgumentDefaultsHelpFormatterNoNone,
        parents=[tmpdir_arg, outputdir_arg])
    enry_parser.set_defaults(handler=install_enry)

    bigartm_parser = subparsers.add_parser(
        "bigartm", help="Install bigartm/bigartm to the current working directory.",
        formatter_class=ArgumentDefaultsHelpFormatterNoNone,
        parents=[tmpdir_arg, outputdir_arg])
    bigartm_parser.set_defaults(handler=install_bigartm)
    dump_parser = subparsers.add_parser(
        "dump", help="Dump a model to stdout.",
        formatter_class=ArgumentDefaultsHelpFormatterNoNone,
        parents=[gcs_arg])
    dump_parser.set_defaults(handler=dump_model)
    dump_parser.add_argument(
        "input", help="Path to the model file, URL or UUID.")

    return parser


def main():
    """
    Creates all the argparse-rs and invokes the function from set_defaults().

    :return: The result of the function from set_defaults().
    """

    parser = get_parser()
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
