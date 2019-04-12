import logging
import os
import pickle
import string
import sys

import keras
import numpy as np
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences

backup = sys.stderr
sys.stderr = open(os.devnull, "w")
sys.stderr.close()
sys.stderr = backup
del backup

MAXLEN = 40
PADDING = "post"


def config_keras():
    import tensorflow as tf
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
    from keras import backend
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    backend.tensorflow_backend.set_session(tf.Session(config=config))


def clear_keras():
    keras.backend.clear_session()


def prepare_single_identifier(identifier, maxlen=MAXLEN, padding=PADDING, mapping=None):
    if mapping is None:
        mapping = dict((c, i + 1) for i, c in enumerate(string.ascii_lowercase))

    # Clean identifier
    clean_id = "".join([char for char in identifier.lower() if char in string.ascii_lowercase])
    if len(clean_id) > MAXLEN:
        clean_id = clean_id[:MAXLEN]
    logging.info("Preprocessed identifier: {}".format(clean_id))

    return np.array([mapping[c] for c in clean_id], dtype='int8'), clean_id


def prepare_input(identifiers, maxlen=MAXLEN, padding=PADDING, mapping=None):

    processed_ids = []
    clean_ids = []
    for identifier in identifiers:
        feat, clean_id = prepare_single_identifier(identifier)
        processed_ids.append(feat)
        clean_ids.append(clean_id)

    processed_ids = pad_sequences(processed_ids, maxlen=maxlen,
                                  padding=padding)

    return processed_ids, clean_ids


def nn_split_ids(identifiers: [str]) -> [str]:
    """
    NN-Based source code identifier splitter
    """
    feats, clean_ids = prepare_input(identifiers)
    output = model.predict(feats)
    output = np.round(output)[:, :, 0]

    splitted_ids = []
    for clean_id, id_output in zip(clean_ids, output):
        splitted_id = ""
        for char, label in zip(clean_id, output):
            if label == 1:
                splitted_ids.append(splitted_id)
                splitted_id = ''
            splitted_id += char
        splitted_ids.append(splitted_id)
    return splitted_ids


def load_model(path="/storage/egor/tmp/v3_fast/rnn.model"):

    model = keras.models.load_model(path, custom_objects={"precision": compute_precision,
                                                          "recall": compute_recall})

    # model.summary()

    return model


def main():
    config_keras()
    model = load_model()

    for line in sys.stdin:
        identifier = line.strip()
