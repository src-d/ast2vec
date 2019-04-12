import logging

from modelforge import Model, register_model
from sourced.ml.models.license import DEFAULT_LICENSE
import keras
from keras.preprocessing.sequence import pad_sequences
import tensorflow
import numpy as np
import string
from keras import backend as K
from sourced.ml.algorithms.id_splitter.nn_model import f1score
import os

MAXLEN = 40
PADDING = "post"


def compute_precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def compute_recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


@register_model
class IdentifierSplitterNN(Model):
    """
    Bi-Directionnal LSTM Model. Splits identifiers without need for a conventional pattern.
    """
    NAME = "id_splitter_nn"
    VENDOR = "source{d}"
    DESCRIPTION = "Model that contains source code identifier splitter BiLSTM weights."
    LICENSE = DEFAULT_LICENSE

    def construct(self, model: "keras.models.Model" = None,
                  session: "tensorflow.Session" = None):
        assert model is not None
        import tensorflow as tf
        if session is None:
            config = tf.ConfigProto()
            tf_session = tf.Session(config=config)
        else:
            backend.tensorflow_backend.set_session(session)
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
        from keras import backend
        config.gpu_options.allow_growth = True

        self._model = model
        return self

    @property
    def model(self):
        """
        Returns the wrapped keras model.
        """
        return self._model

    def clear_keras(self):
        """
        Clears the keras session used to run the model
        """
        keras.backend.clear_session()

    def _generate_tree(self) -> dict:
        return {
                "config": self._model.get_config(),
                "weights": self._model.get_weights()
                }

    def _load_tree(self, tree: dict):
        from keras.models import Model

        model = Model.from_config(tree["config"])
        model.set_weights(tree["weights"])
        self.construct(model=model)

    def _prepare_single_identifier(self, identifier, maxlen=MAXLEN, padding=PADDING, mapping=None):
        if mapping is None:
            mapping = dict((c, i + 1) for i, c in enumerate(string.ascii_lowercase))

        # Clean identifier
        clean_id = "".join([char for char in identifier.lower() if char in string.ascii_lowercase])
        if len(clean_id) > MAXLEN:
            clean_id = clean_id[:MAXLEN]
        logging.info("Preprocessed identifier: {}".format(clean_id))
        return np.array([mapping[c] for c in clean_id], dtype='int8'), clean_id

    def prepare_input(self, identifiers: [str], maxlen: int = MAXLEN, padding: int = PADDING,
                      mapping=None):

        processed_ids = []
        clean_ids = []
        for identifier in identifiers:
            feat, clean_id = self._prepare_single_identifier(
                identifier, maxlen=maxlen, padding=padding)
            processed_ids.append(feat)
            clean_ids.append(clean_id)

        processed_ids = pad_sequences(processed_ids, maxlen=maxlen,
                                      padding=padding)

        return processed_ids, clean_ids

    def load_model_file(self, path="./id_splitter_rnn.model"):

        self._model = keras.models.load_model(path, custom_objects={"precision": compute_precision,
                                                                    "recall": compute_recall,
                                                                    "f1score": f1score})

    def __call__(self, tokens: [str]) -> [[str]]:
        """
        Splits a lists of tokens using the model.
        """
        feats, clean_ids = self.prepare_input(tokens)
        output = self._model.predict(feats)
        output = np.round(output)[:, :, 0]
        splitted_ids = []
        for clean_id, id_output in zip(clean_ids, output):
            splitted_id = ""
            for char, label in zip(clean_id, id_output):
                if label == 1:
                    splitted_ids.append(splitted_id)
                    splitted_id = ''
                splitted_id += char
            splitted_ids.append(splitted_id)
        return splitted_ids

    def __del__(self):
        self.clear_keras()
