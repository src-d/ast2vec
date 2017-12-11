import argparse
import logging
import os
import sys

from modelforge.logs import setup_logging
from sourced.ml.algorithms.id_embedding import preprocess as preprocess_id2vec, run_swivel, \
    postprocess as postprocess_id2vec, swivel
from sourced.ml.models.bow import bow2vw_entry
from sourced.ml.utils import install_bigartm, bigartm2asdf_entry
from sourced.ml.utils import dump_model
from sourced.ml.utils import projector_entry


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
    repos2input_arg = one_arg_parser(
        "input", nargs="+", help="List of repositories and/or files with list of repositories.")
    output_dir_arg_default = one_arg_parser(
        "-o", "--output", required=True, help="Output directory.")
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
