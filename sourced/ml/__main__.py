import argparse
import logging
import os
import sys

from modelforge.logs import setup_logging

from sourced.ml.extractors import IdentifierDistance
from sourced.ml.cmd_entries import bigartm2asdf_entry, dump_model, projector_entry, bow2vw_entry, \
    run_swivel, postprocess_id2vec, preprocess_id2vec, repos2coocc_entry, repos2df_entry, \
    repos2ids_entry, repos2bow_entry, repos2roles_and_ids_entry, repos2id_distance_entry, \
    repos2id_sequence_entry
from sourced.ml.cmd_entries.args import add_df_args, add_feature_args, \
    add_vocabulary_size_arg, add_repo2_args, add_split_stem_arg, \
    ArgumentDefaultsHelpFormatterNoNone, add_bow_args
from sourced.ml.cmd_entries.run_swivel import mirror_tf_args
from sourced.ml.utils import install_bigartm


def get_parser() -> argparse.ArgumentParser:
    """
    Creates the cmdline argument parser.
    """

    parser = argparse.ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatterNoNone)
    parser.add_argument("--log-level", default="INFO",
                        choices=logging._nameToLevel,
                        help="Logging verbosity.")
    # Create and construct subparsers
    subparsers = parser.add_subparsers(help="Commands", dest="command")

    def add_parser(name, help_message):
        return subparsers.add_parser(
            name, help=help_message, formatter_class=ArgumentDefaultsHelpFormatterNoNone)

    # ------------------------------------------------------------------------
    repos2bow_parser = add_parser(
        "repos2bow", "Convert source code to the bag-of-words model.")
    repos2bow_parser.set_defaults(handler=repos2bow_entry)
    add_df_args(repos2bow_parser)
    add_repo2_args(repos2bow_parser)
    add_feature_args(repos2bow_parser)
    add_bow_args(repos2bow_parser)
    # ------------------------------------------------------------------------
    repos2df_parser = add_parser(
        "repos2df", "Calculate document frequencies of features extracted from source code.")
    repos2df_parser.set_defaults(handler=repos2df_entry)
    add_df_args(repos2df_parser)
    add_repo2_args(repos2df_parser)
    add_feature_args(repos2df_parser)
    # ------------------------------------------------------------------------
    repos2ids_parser = subparsers.add_parser(
        "repos2ids", help="Convert source code to a bag of identifiers.")
    repos2ids_parser.set_defaults(handler=repos2ids_entry)
    add_repo2_args(repos2ids_parser)
    add_split_stem_arg(repos2ids_parser)
    repos2ids_parser.add_argument(
        "-o", "--output", required=True,
        help="[OUT] output path to the CSV file with identifiers.")
    repos2ids_parser.add_argument(
        "--idfreq", action="store_true",
        help="Adds identifier frequencies to the output CSV file."
             "num_repos is the number of repositories where the identifier appears in."
             "num_files is the number of files where the identifier appears in."
             "num_occ is the total number of occurences of the identifier.")
    # ------------------------------------------------------------------------
    repos2coocc_parser = add_parser(
        "repos2coocc", "Convert source code to the sparse co-occurrence matrix of identifiers.")
    repos2coocc_parser.set_defaults(handler=repos2coocc_entry)
    add_df_args(repos2coocc_parser)
    add_repo2_args(repos2coocc_parser)
    add_split_stem_arg(repos2coocc_parser)
    repos2coocc_parser.add_argument(
        "-o", "--output", required=True,
        help="[OUT] Path to the Cooccurrences model.")
    # ------------------------------------------------------------------------
    repos2roles_and_ids = add_parser(
        "repos2roles_ids", "Converts a UAST to a list of pairs, where pair is a role and "
        "identifier. Role is merged generic roles where identifier was found.")
    repos2roles_and_ids.set_defaults(handler=repos2roles_and_ids_entry)
    add_repo2_args(repos2roles_and_ids)
    add_split_stem_arg(repos2roles_and_ids)
    repos2roles_and_ids.add_argument(
        "-o", "--output", required=True,
        help="[OUT] Path to the directory where spark should store the result. "
             "Inside the direcory you find result is csv format, status file and sumcheck files.")
    # ------------------------------------------------------------------------
    repos2identifier_distance = add_parser(
        "repos2id_distance", "Converts a UAST to a list of identifier pairs "
                             "and distance between them.")
    repos2identifier_distance.set_defaults(handler=repos2id_distance_entry)
    add_repo2_args(repos2identifier_distance)
    add_split_stem_arg(repos2identifier_distance)
    repos2identifier_distance.add_argument(
        "-t", "--type", required=True, choices=IdentifierDistance.DistanceType.All,
        help="Distance type.")
    repos2identifier_distance.add_argument(
        "--max-distance", default=IdentifierDistance.DEFAULT_MAX_DISTANCE,
        help="Maximum distance to save.")
    repos2identifier_distance.add_argument(
        "-o", "--output", required=True,
        help="[OUT] Path to the directory where spark should store the result. "
             "Inside the direcory you find result is csv format, status file and sumcheck files.")
    # ------------------------------------------------------------------------
    repos2id_sequence = add_parser(
        "repos2id_sequence", "Converts a UAST to sequence of identifiers sorted by "
                             "order of appearance.")
    repos2id_sequence.set_defaults(handler=repos2id_sequence_entry)
    add_repo2_args(repos2id_sequence)
    add_split_stem_arg(repos2id_sequence)
    repos2id_sequence.add_argument(
        "--skip-docname", default=False, action="store_true",
        help="Do not save document name in CSV file, only identifier sequence.")
    repos2id_sequence.add_argument(
        "-o", "--output", required=True,
        help="[OUT] Path to the directory where spark should store the result. "
             "Inside the direcory you find result is csv format, status file and sumcheck files.")
    # ------------------------------------------------------------------------
    preproc_parser = add_parser(
        "id2vec_preproc", "Convert a sparse co-occurrence matrix to the Swivel shards.")
    preproc_parser.set_defaults(handler=preprocess_id2vec)
    add_vocabulary_size_arg(preproc_parser)
    preproc_parser.add_argument("-s", "--shard-size", default=4096, type=int,
                                help="The shard (submatrix) size.")
    preproc_parser.add_argument(
        "--docfreq", default=None,
        help="[IN] Path to the pre-calculated document frequencies in asdf format "
             "(DF in TF-IDF).")
    preproc_parser.add_argument(
        "-i", "--input",
        help="Concurrence model produced by repos2coocc.")
    preproc_parser.add_argument("-o", "--output", required=True, help="Output directory.")
    # ------------------------------------------------------------------------
    train_parser = add_parser(
        "id2vec_train", "Train identifier embeddings using Swivel.")
    train_parser.set_defaults(handler=run_swivel)
    mirror_tf_args(train_parser)
    # ------------------------------------------------------------------------
    id2vec_postproc_parser = add_parser(
        "id2vec_postproc",
        "Combine row and column embeddings produced by Swivel and write them to an .asdf.")
    id2vec_postproc_parser.set_defaults(handler=postprocess_id2vec)
    id2vec_postproc_parser.add_argument(
        "-i", "--swivel-data", required=True,
        help="Folder with swivel batches input data. You can get it using repos2coocc subcommand.")
    id2vec_postproc_parser.add_argument(
        "-o", "--output", required=True,
        help="Output directory for embedding data.")
    # ------------------------------------------------------------------------
    id2vec_project_parser = add_parser(
        "id2vec_project", "Present id2vec model in Tensorflow Projector.")
    id2vec_project_parser.set_defaults(handler=projector_entry)
    id2vec_project_parser.add_argument("-i", "--input", required=True,
                                       help="id2vec model to present.")
    id2vec_project_parser.add_argument("-o", "--output", required=True,
                                       help="Projector output directory.")
    id2vec_project_parser.add_argument("--docfreq", help="docfreq model to pick the most "
                                                         "significant identifiers.")
    id2vec_project_parser.add_argument("--no-browser", action="store_true",
                                       help="Do not open the browser.")
    # ------------------------------------------------------------------------
    bow2vw_parser = add_parser(
        "bow2vw", "Convert a bag-of-words model to the dataset in Vowpal Wabbit format.")
    bow2vw_parser.set_defaults(handler=bow2vw_entry)
    bow2vw_parser.add_argument(
        "--bow", help="URL or path to a bag-of-words model.")
    bow2vw_parser.add_argument(
        "--id2vec", help="URL or path to the identifier embeddings.")
    bow2vw_parser.add_argument(
        "-o", "--output", required=True, help="Path to the output file.")
    # ------------------------------------------------------------------------
    bigartm_postproc_parser = add_parser(
        "bigartm2asdf", "Convert a human-readable BigARTM model to Modelforge format.")
    bigartm_postproc_parser.set_defaults(handler=bigartm2asdf_entry)
    bigartm_postproc_parser.add_argument("input")
    bigartm_postproc_parser.add_argument("output")
    # ------------------------------------------------------------------------
    bigartm_parser = add_parser(
        "bigartm", "Install bigartm/bigartm to the current working directory.")
    bigartm_parser.set_defaults(handler=install_bigartm)
    bigartm_parser.add_argument(
        "--tmpdir", help="Store intermediate files in this directory instead of /tmp.")
    bigartm_parser.add_argument("--output", default=os.getcwd(), help="Output directory.")
    # ------------------------------------------------------------------------
    dump_parser = add_parser("dump", "Dump a model to stdout.")
    dump_parser.set_defaults(handler=dump_model)
    dump_parser.add_argument(
        "input", help="Path to the model file, URL or UUID.")
    dump_parser.add_argument("--gcs", default=None, dest="gcs_bucket",
                             help="GCS bucket to use.")

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
