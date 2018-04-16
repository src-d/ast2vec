import argparse
import json

from sourced.ml import extractors
from sourced.ml.transformers import Moder


class ArgumentDefaultsHelpFormatterNoNone(argparse.ArgumentDefaultsHelpFormatter):
    """
    Pretty formatter of help message for arguments.
    It adds default value to the end if it is not None.
    """
    def _get_help_string(self, action):
        if action.default is None:
            return action.help
        return super()._get_help_string(action)


def add_split_stem_arg(my_parser: argparse.ArgumentParser):
    my_parser.add_argument(
        "--split", action="store_true",
        help="Split identifiers based on special characters or case changes. For example split "
             "'ThisIs_token' to ['this', 'is', 'token'].")


def add_vocabulary_size_arg(my_parser: argparse.ArgumentParser):
    my_parser.add_argument(
        "-v", "--vocabulary-size", default=10000000, type=int,
        help="The maximum vocabulary size.")


def add_extractor_args(my_parser: argparse.ArgumentParser):
    my_parser.add_argument(
        "-r", "--repositories", required=True,
        help="The path to the repositories.")
    my_parser.add_argument(
        "-l", "--languages", required=True, nargs="+", choices=(
            "Java", "Python", "JavaScript", "Ruby", "Bash"),
        help="The programming languages to analyse.")


def add_repo2_args(my_parser: argparse.ArgumentParser, quant=True):
    add_extractor_args(my_parser)
    my_parser.add_argument(
        "--graph", help="Write the tree in Graphviz format to this file.")
    my_parser.add_argument(
        "--min-docfreq", default=1, type=int,
        help="The minimum document frequency of each feature.")
    add_vocabulary_size_arg(my_parser)
    my_parser.add_argument(
        "--docfreq", required=True,
        help="[OUT] The path to the OrderedDocumentFrequencies model.")
    if quant:
        my_parser.add_argument(
            "--quant", help="[OUT] The path to the QuantizationLevels model.")


def add_feature_args(my_parser: argparse.ArgumentParser, required=True):
    my_parser.add_argument("-x", "--mode", choices=Moder.Options.__all__,
                           default="file", help="What to select for analysis.")
    my_parser.add_argument(
        "-f", "--feature", nargs="+",
        choices=[ex.NAME for ex in extractors.__extractors__.values()],
        required=required, help="The feature extraction scheme to apply.")
    for ex in extractors.__extractors__.values():
        for opt, val in ex.OPTS.items():
            my_parser.add_argument(
                "--%s-%s" % (ex.NAME, opt), default=val, type=json.loads,
                help="%s's kwarg" % ex.__name__)
