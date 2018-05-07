import argparse
from datetime import datetime as dt
import logging
import pickle
import random
import os
import string
import tarfile
import warnings

import numpy as np
try:
    import keras
    from keras import backend as K
    from keras.callbacks import CSVLogger, TensorBoard, ModelCheckpoint, LearningRateScheduler
    from keras.layers import BatchNormalization, Concatenate, Conv1D, CuDNNGRU, CuDNNLSTM, Dense, \
        Embedding, GRU, Input, LSTM, TimeDistributed
    from keras.models import Model
    from keras.preprocessing.sequence import pad_sequences
    import tensorflow as tf
except ImportError as e:
    warnings.warn("Tensorflow or/and Keras are not installed, dependent functionality is "
                  "unavailable.")


# Common default parameters
MAXLEN = 40  # max length of sequence
PADDING = "post"  # add padding values after input
LOSS = "binary_crossentropy"
EPOCHS = 10
BATCH_SIZE = 500
VAL_BATCH_SIZE = 2000
START_LR = 0.001
FINAL_LR = 0.00001
DEFAULT_DEVICES = "0"
RANDOM_SEED = 1989
SAMPLES_BEFORE_REPORT = 5 * 10 ** 6
EPSILON = 10 ** -8
DEFAULT_THRESHOLD = 0.5  # threshold that is used to binarize predictions of the model
TEST_SIZE = 0.2  # fraction of dataset to use as test

# CSV default parameters
TOKEN_COL = 0
TOKEN_SPLIT_COL = 1

# RNN default parameters
DEFAULT_RNN_TYPE = "CuDNNLSTM"
RNN_TYPES = ("GRU", "LSTM", "CuDNNLSTM", "CuDNNGRU")

# CNN default parameters
FILTERS = "64,32,16,8"
KERNEL_SIZES = "2,4,8,16"
DIM_REDUCTION = 32

METRICS = ["accuracy"]


def register_metric(metric):
    assert isinstance(metric, str) or callable(metric)
    METRICS.append(metric)
    return metric


def prepare_input_emb(maxlen, n_uniq):
    """
    One-hot encoding of characters
    :param maxlen: maximum length of input sequence
    :param n_uniq: number of unique characters
    :return: tensor for input, one-hot character embeddings
    """
    char_seq = Input((maxlen,))
    emb = Embedding(input_dim=n_uniq + 1, output_dim=n_uniq + 1, input_length=maxlen,
                    mask_zero=False, weights=[np.eye(n_uniq + 1)], trainable=False)(char_seq)
    return char_seq, emb


def add_output_layer(input_layer):
    """
    Output layer has 1 sigmoid per character that should predict if there's a space before char
    :param input_layer: hidden layer before output layer
    :return: layer
    """
    norm_input = BatchNormalization()(input_layer)
    return TimeDistributed(Dense(1, activation="sigmoid"))(norm_input)


def add_rnn(X, units=128, rnn_layer=None, dev0="/gpu:0",
            dev1="/gpu:1"):
    """
    Add a RNN layer according to parameters.
    :param X: input layer
    :param units: number of neurons in layer
    :param rnn_layer: type of RNN layer
    :param dev0: device that will be used for forward pass of RNN and concatenation
    :param dev1: device that will be used for backward pass
    :return: layer
    """
    # select RNN layer
    rnn_layer_mapping = {"GRU": GRU, "LSTM": LSTM, "CuDNNLSTM": CuDNNLSTM, "CuDNNGRU": CuDNNGRU}

    if rnn_layer is None:
        rnn_layer = CuDNNLSTM
    elif isinstance(rnn_layer, str):
        rnn_layer = rnn_layer_mapping[rnn_layer]

    # add forward & backward RNN
    with tf.device(dev0):
        forward_gru = rnn_layer(units=units, return_sequences=True)(X)
    with tf.device(dev1):
        backward_gru = rnn_layer(units=units, return_sequences=True, go_backwards=True)(X)

    # concatenate
    with tf.device(dev1):
        bidi_gru = Concatenate(axis=-1)([forward_gru, backward_gru])
    return bidi_gru


def prepare_rnn_model(args: argparse.ArgumentParser):
    """
    Construct a RNN model according to given arguments.
    :param args: arguments should contain: num_chars, length, filters, dim_reduction, stack,
                 kernel_sizes, optimizer, devices
    :return: compiled model
    """
    # extract required arguments
    n_uniq = args.num_chars
    maxlen = args.length

    units = args.neurons
    stack = args.stack
    optimizer = args.optimizer
    rnn_layer = args.type
    dev0, dev1 = prepare_devices(args)

    if rnn_layer is None:
        rnn_layer = DEFAULT_RNN_TYPE

    # prepare model
    with tf.device(dev0):
        char_seq, hid_layer = prepare_input_emb(maxlen, n_uniq)

        # stack BiDi-RNN
        for i in range(stack):
            hid_layer = add_rnn(hid_layer, units=units, rnn_layer=rnn_layer, dev0=dev0, dev1=dev1)

        output = add_output_layer(hid_layer)

    # prepare optimizer
    opt = keras.optimizers.get(optimizer)(lr=args.rate)

    # compile model
    model = Model(inputs=char_seq, outputs=output)
    model.compile(optimizer=opt, loss=LOSS, metrics=METRICS)
    return model


def add_conv(X, filters=[64, 32, 16, 8], kernel_sizes=[2, 4, 8, 16], output_n_filters=32):
    """
    Build a single convolutional layer.
    :param X: previous layer
    :param filters: number of filter for each kernel size
    :param kernel_sizes: list of kernel sizes
    :param output_n_filters: number of 1d output filters
    :return: layer
    """
    # normalization of input
    X = BatchNormalization()(X)

    # add convolutions
    convs = []

    for n_filters, kern_size in zip(filters, kernel_sizes):
        conv = Conv1D(filters=n_filters, kernel_size=kern_size, padding="same",
                      activation="relu")
        convs.append(conv(X))

    # concatenate all convolutions
    conc = Concatenate(axis=-1)(convs)
    conc = BatchNormalization()(conc)

    # dimensionality reduction
    conv = Conv1D(filters=output_n_filters, kernel_size=1, padding="same", activation="relu")
    return conv(conc)


def prepare_cnn_model(args: argparse.ArgumentParser):
    """
    Construct a CNN model according to given arguments.
    :param args: arguments should contain: num_chars, length, filters, dim_reduction, stack,
                 kernel_sizes, optimizer, devices
    :return: compiled model
    """
    # extract required arguments
    n_uniq = args.num_chars
    maxlen = args.length

    def to_list(params):
        """
        Convert string parameters to list.
        :param params: string that contains integer parameters separated by comma
        :return: list of integers
        """
        return list(map(int, params.split(",")))

    filters = to_list(args.filters)
    output_n_filters = args.dim_reduction
    stack = args.stack
    kernel_sizes = to_list(args.kernel_sizes)
    optimizer = args.optimizer
    device, _ = prepare_devices(args)

    # prepare model
    with tf.device(device):
        char_seq, hid_layer = prepare_input_emb(maxlen, n_uniq)

        # stack CNN
        for _ in range(stack):
            hid_layer = add_conv(hid_layer, filters=filters, kernel_sizes=kernel_sizes,
                                 output_n_filters=output_n_filters)

        output = add_output_layer(hid_layer)

    # compile model
    model = Model(inputs=char_seq, outputs=output)

    model.compile(optimizer=optimizer, loss=LOSS, metrics=METRICS)
    return model


def set_random_seed(seed):
    """
    Fix random seed for reproducibility.
    :param seed: seed value
    """
    np.random.seed(seed)
    random.seed(seed)
    tf.set_random_seed(seed)


@register_metric
def precision(y_true, y_pred):
    """
    Precision metric. Only computes a batch-wise average of recall.
    :param y_true: tensor
    :param y_pred: tensor
    :return: tensor
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


@register_metric
def recall(y_true, y_pred):
    """
    Recall metric. Only computes a batch-wise average of recall.
    :param y_true: tensor
    :param y_pred: tensor
    :return: tensor
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


@register_metric
def f1score(y_true, y_pred):
    """
    F1 score. Only computes a batch-wise average of recall.
    :param y_true: tensor
    :param y_pred: tensor
    :return: tensor
    """
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * prec * rec / (prec + rec + K.epsilon())


def to_binary(mat, threshold):
    """
    Helper function to binarize matrix
    :param mat: matrix or array
    :param threshold: if value >= threshold than it will be 1, else 0
    :return: binarized matrix
    """
    mask = mat >= threshold
    mat[mask] = 1
    mat[np.logical_not(mask)] = 0
    return mat


def precision_np(y_true, y_pred, epsilon=EPSILON):
    """
    Precision metric.
    :param y_true: ground truth labels - expect binary values
    :param y_pred: predicted labels - expect binary values
    :param epsilon: to avoid division by zero
    :return: precision
    """
    true_positives = np.sum(y_true * y_pred)
    predicted_positives = np.sum(y_pred)
    return true_positives / (predicted_positives + epsilon)


def recall_np(y_true, y_pred, epsilon=EPSILON):
    """
    Compute recall metric.
    :param y_true: matrix with ground truth labels - expect binary values
    :param y_pred: matrix with predicted labels - expect binary values
    :param epsilon: added to denominator to avoid division by zero
    :return: recall
    """
    true_positives = np.sum(y_true * y_pred)
    possible_positives = np.sum(y_true)
    return true_positives / (possible_positives + epsilon)


def report(model, X, y, batch_size=VAL_BATCH_SIZE, threshold=DEFAULT_THRESHOLD, epsilon=EPSILON):
    """
    Prepare report for `model` on data `X` & `y`. It prints precision, recall, F1 score.
    :param model: model to apply
    :param X: features
    :param y: labels (expected binary labels)
    :param batch_size: batch size that will be used or prediction
    :param threshold: threshold to binarize predictions
    :param epsilon: added to denominator to avoid division by zero
    """
    log = logging.getLogger("quality-reporter")

    # predict & skip the last dimension & binarize
    predictions = model.predict(X, batch_size=batch_size, verbose=1)[:, :, 0]
    predictions = to_binary(predictions, threshold)

    # report
    pr = precision_np(y[:, :, 0], predictions, epsilon=epsilon)
    rec = recall_np(y[:, :, 0], predictions, epsilon=epsilon)
    f1 = 2 * pr * rec / (pr + rec + epsilon)
    log.info("precision: {:.3f}, recall: {:.3f}, f1: {:.3f}".format(pr, rec, f1))


def load_model(path, add_precision: bool=True, add_recall: bool=True, add_f1score: bool=True,
               print_fn=None):
    """
    Load saved model and print summary.
    :param path: location of the saved model
    :param add_precision: if precision should be added to custom objects
    :param add_recall: if recall should be added to custom objects
    :param add_f1score: if f1score should be added to custom objects
    :param print_fn: print function to use. If None than print will be used
    :return: model
    """
    custom_obj = {}
    if add_precision:
        custom_obj["precision"] = precision
    if add_recall:
        custom_obj["recall"] = recall
    if add_f1score:
        custom_obj["f1score"] = f1score

    model = keras.models.load_model(path, custom_objects=custom_obj)

    if print_fn is None:
        print_fn = print

    model.summary(print_fn=print_fn)

    return model


def load_features(path):
    """
    Load features. It should be pickled dictionary with 4 keys: 'x_tr', 'x_t', 'y_tr', 'y_t'.
    :param path: location of pickle file
    :return: x_tr, x_t, y_tr, y_t
    """
    with open(path, "rb") as f:
        data = pickle.load(f)

    return data["x_tr"], data["x_t"], data["y_tr"], data["y_t"]


def config_keras():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    K.tensorflow_backend.set_session(tf.Session(config=config))


def prepare_train_generator(x, y, batch_size=500):
    def xy_generator():
        while True:
            n_batches = x.shape[0] // batch_size
            for i in range(n_batches):
                st = i * batch_size
                end = (i + 1) * batch_size
                yield (x[st:end], y[st:end])
    return xy_generator()


def make_lr_scheduler(lr=0.001, final_lr=0.00001, n_epochs=10, verbose=1):
    """
    Prepare learning rate scheduler to change learning rate during the training.
    :param lr: initial learning rate
    :param final_lr: final learning rate
    :param n_epochs: number of epochs
    :param verbose: verbosity
    :return: LearningRateScheduler with linear schedul of learning rates
    """
    delta = (lr - final_lr) / (n_epochs + 1)

    def schedule(epoch):
        assert 0 <= epoch < n_epochs
        return lr - delta * epoch

    return LearningRateScheduler(schedule=schedule, verbose=verbose)


def prepare_devices(args: argparse.ArgumentParser):
    """
    Extract devices from arguments
    :param args: arguments
    :return: splitted devices
    """
    devices = args.devices.split(",")
    if len(devices) == 2:
        dev0, dev1 = ("/gpu:" + dev for dev in devices)
    elif len(devices) == 1:
        if int(devices[0]) != -1:
            dev0 = dev1 = "/gpu:" + args.devices
        else:
            dev0 = dev1 = "/cpu:0"
    else:
        raise ValueError("Expected 1 or 2 devices but got {} from args.devices argument {}"
                         .format(len(devices), args.devices))

    return dev0, dev1


def prepare_callbacks(output_dir):
    """
    Prepare logging, tensorboard, model checkpoint callbacks and store their outputs at output_dir.
    :param output_dir: location of directory to store results.
    :return: list of callbacks
    """
    time = dt.now().strftime("%y%m%d-%H%M")
    log_dir = os.path.join(output_dir, "tensorboard" + time)
    logging.info("Tensorboard directory: {}".format(log_dir))
    tensorboard = TensorBoard(log_dir=log_dir, batch_size=1000, write_images=True,
                              write_graph=True)
    csv_loc = os.path.join(output_dir, "csv_logger_" + time + ".txt")
    logging.info("CSV logs: {}".format(csv_loc))
    csv_logger = CSVLogger(csv_loc)

    filepath = os.path.join(output_dir, "best_" + time + ".model")
    model_saver = ModelCheckpoint(filepath, monitor='val_recall', verbose=1, save_best_only=True,
                                  mode='max')
    return [tensorboard, csv_logger, model_saver]


def train_parameters(batch_size, samples_per_epoch, n_samples, epochs):
    """
    Helper function to split huge dataset into smaller one to make reports more frequently
    :param batch_size: batch size
    :param samples_per_epoch: number of samples per mini-epoch or before report
    :param n_samples: total number of samples
    :param epochs: number epochs over full dataset
    :return: number of steps per epoch (should be used with generator) and number of sub-epochs
             where during sub-epoch only samples_per_epoch will be generated
    """
    steps_per_epoch = samples_per_epoch // batch_size
    n_epochs = np.ceil(epochs * n_samples / samples_per_epoch)
    return steps_per_epoch, n_epochs


def prepare_features(csv_loc, use_header=True, token_col=TOKEN_COL, maxlen=MAXLEN, mode="r",
                     token_split_col=TOKEN_SPLIT_COL, test_size=0.2, padding=PADDING):
    log = logging.getLogger("id-splitter-prep")

    # read data from file
    log.info("Reading data from CSV...")
    identifiers = []
    with tarfile.open(csv_loc, mode=mode, encoding="utf-8") as f:
        assert len(f.members) == 1, "Expect one archived file"
        content = f.extractfile(f.members[0])
        if not use_header:
            content.readline()
        for line in content:
            parts = line.decode("utf-8").strip().split(",")
            if len(parts[token_col]) <= maxlen:
                identifiers.append(parts[token_split_col])
    np.random.shuffle(identifiers)
    log.info("Number of identifiers after filtering: {}.".format(len(identifiers)))

    # convert identifiers into character indices and labels
    log.info("Converting identifiers to character indices...")

    char2ind = dict((c, i + 1) for i, c in enumerate(sorted(string.ascii_lowercase)))

    char_id_seq = []
    splits = []
    for identifier in identifiers:
        # iterate through identifier and convert to array of char indices & boolean split array
        index_arr = []
        split_arr = []
        skip_char = False
        for char in identifier.strip():
            if char in char2ind:
                index_arr.append(char2ind[char])
                if skip_char:
                    skip_char = False
                    continue
                split_arr.append(0)
            else:
                # space
                split_arr.append(1)
                skip_char = True
        # sanity check
        assert len(index_arr) == len(split_arr)
        char_id_seq.append(index_arr)
        splits.append(split_arr)

    # train/test splitting
    log.info("Train/test splitting...")
    n_train = int((1 - test_size) * len(char_id_seq))
    x_tr = char_id_seq[:n_train]
    x_t = char_id_seq[n_train:]
    y_tr = splits[:n_train]
    y_t = splits[n_train:]
    log.info("Number of train samples: {}, number of test samples: {}.".format(len(x_tr),
                                                                               len(x_t)))

    # pad sequence
    log.info("Padding of the sequences...")
    x_tr = pad_sequences(x_tr, maxlen=maxlen, padding=padding)
    x_t = pad_sequences(x_t, maxlen=maxlen, padding=padding)
    y_tr = pad_sequences(y_tr, maxlen=maxlen, padding=padding)
    y_t = pad_sequences(y_t, maxlen=maxlen, padding=padding)

    return x_tr, x_t, y_tr[:, :, None], y_t[:, :, None]


def pipeline(args: argparse.ArgumentParser, prepare_model):
    log = logging.getLogger("id-splitter-train")
    config_keras()
    set_random_seed(args.seed)

    # prepare features
    x_tr, x_t, y_tr, y_t = prepare_features(csv_loc=args.input, use_header=args.csv_header,
                                            token_col=args.csv_token, maxlen=args.length,
                                            token_split_col=args.csv_token_split,
                                            test_size=args.test_size, padding=args.padding)

    # prepare train generator
    steps_per_epoch, n_epochs = train_parameters(batch_size=args.batch_size,
                                                 samples_per_epoch=args.samples_before_report,
                                                 n_samples=x_tr.shape[0], epochs=args.epochs)

    train_gen = prepare_train_generator(x=x_tr, y=y_tr, batch_size=args.batch_size)

    # prepare test generator
    validation_steps, _ = train_parameters(batch_size=args.val_batch_size,
                                           samples_per_epoch=x_t.shape[0],
                                           n_samples=x_t.shape[0], epochs=args.epochs)
    test_gen = prepare_train_generator(x=x_t, y=y_t, batch_size=args.val_batch_size)

    # initialize model
    model = prepare_model(args)
    log.info("Model summary:")
    model.summary(print_fn=log.info)

    # callbacks
    callbacks = prepare_callbacks(args.output)
    if not args.skip_lr_scheduler:
        lr_scheduler = make_lr_scheduler(lr=args.rate, final_lr=args.final_rate, n_epochs=n_epochs)
        callbacks.append(lr_scheduler)

    # train
    history = model.fit_generator(train_gen, steps_per_epoch=steps_per_epoch,
                                  validation_data=test_gen, validation_steps=validation_steps,
                                  callbacks=callbacks, epochs=n_epochs)

    # report quality on test dataset
    report(model, X=x_t, y=y_t, batch_size=args.val_batch_size)

    # save model & history
    with open(os.path.join(args.output, "model.history"), "wb") as f:
        pickle.dump(history.history, f)
    model.save(os.path.join(args.output, "last.model"))
    log.info("Completed!")
