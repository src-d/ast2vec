import argparse
import string

from sourced.ml.cmd_entries import ArgumentDefaultsHelpFormatterNoNone
from sourced.ml.algorithms.id_splitter import EPOCHS, BATCH_SIZE, MAXLEN, TEST_SIZE, START_LR, \
    PADDING, FINAL_LR, SAMPLES_BEFORE_REPORT, VAL_BATCH_SIZE, RANDOM_SEED, DEFAULT_DEVICES, \
    TOKEN_COL, TOKEN_SPLIT_COL, DEFAULT_RNN_TYPE, RNN_TYPES, FILTERS, KERNEL_SIZES, \
    DIM_REDUCTION, prepare_rnn_model, prepare_cnn_model, pipeline
from sourced.ml.utils.engine import pause


@pause
def rnn_pipeline(args):
    return pipeline(args, prepare_model=prepare_rnn_model)


@pause
def cnn_pipeline(args):
    return pipeline(args, prepare_model=prepare_cnn_model)


def add_id_splitter_arguments(parser: argparse.ArgumentParser):
    subparsers = parser.add_subparsers(help="Identifier splitter", dest="id-splitter")

    def add_parser(name, help_message):
        return subparsers.add_parser(
            name, help=help_message, formatter_class=ArgumentDefaultsHelpFormatterNoNone)
    # common arguments for CNN/RNN models
    parser.add_argument("-i", "--input", help="Path to CSV file.", required=True)
    parser.add_argument("-e", "--epochs", type=int, default=EPOCHS, help="Number of epochs.")
    parser.add_argument("-b", "--batch-size", type=int, default=BATCH_SIZE, help="Batch size.")
    parser.add_argument("-l", "--length", type=int, default=MAXLEN, help="RNN sequence length.")
    parser.add_argument("-o", "--output", required=True,
                        help="Path to the output folder to store results.")
    parser.add_argument("-t", "--test-size", type=float, default=TEST_SIZE,
                        help="Fraction of dataset to use as test.")
    parser.add_argument("-p", "--padding", default=PADDING, choices=("pre", "post"),
                        help="Pad either before or after each sequence.")
    # TODO: list available optimizers from keras and add their arguments
    parser.add_argument("--optimizer", default="Adam", choices=("RMSprop", "Adam"),
                        help="Optimizer to apply.")
    parser.add_argument("--rate", default=START_LR, type=float,
                        help="Initial learning rate of optimizer.")
    parser.add_argument("--final-rate", default=FINAL_LR, type=float,
                        help="Final learning rate of optimizer.")
    parser.add_argument("--samples-before-report", type=int, default=SAMPLES_BEFORE_REPORT,
                        help="Number of samples before show validation report & other updates.")
    parser.add_argument("--val-batch-size", type=int, default=VAL_BATCH_SIZE,
                        help="Batch size for validation - can be increased to speed up pipeline. "
                             "But don't forget about memory consumption.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed.")
    parser.add_argument("--num-chars", type=int, default=len(string.ascii_lowercase),
                        help="Number of unique characters.")
    parser.add_argument("--devices", default=DEFAULT_DEVICES, help="Device(s) to use. '-1' means "
                                                                   "CPU.")
    parser.add_argument("--csv-token", default=TOKEN_COL, help="Column with raw token.")
    parser.add_argument("--csv-token-split", default=TOKEN_SPLIT_COL, help="Column with splitted "
                                                                           "token.")
    header_help = "Treat first line of CSV as normal line instead of header with column names."
    parser.add_argument("--csv-header", action="store_true", help=header_help)

    # RNN specific arguments
    rnn_parser = add_parser("rnn", "Train RNN model to split identifiers.")

    rnn_parser.set_defaults(handler=rnn_pipeline)
    rnn_parser.add_argument("-t", "--type", default=DEFAULT_RNN_TYPE,
                            choices=RNN_TYPES,
                            help="Recurrent layer type to use.")
    rnn_parser.add_argument("-n", "--neurons", default=256, type=int,
                            help="Number of neurons on each layer.")
    rnn_parser.add_argument("-s", "--stack", default=2, type=int,
                            help="Number of BiDi-RNN stacked on each other.")

    # CNN specific arguments
    cnn_parser = add_parser("cnn", "Train CNN model to split identifiers.")

    cnn_parser.set_defaults(handler=cnn_pipeline)
    cnn_parser.add_argument("-f", "--filters", default=FILTERS,
                            help="Number of filters for each kernel size.")
    cnn_parser.add_argument("-s", "--stack", default=3, type=int,
                            help="Number of CNN layers stacked on each other.")
    cnn_parser.add_argument("-k", "--kernel-sizes", default=KERNEL_SIZES,
                            help="Sizes for sliding windows.")
    cnn_parser.add_argument("--dim-reduction", default=DIM_REDUCTION, type=int,
                            help="Number of 1-d kernels to reduce dimensionality after each "
                                 "layer.")
