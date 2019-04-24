import logging
import os
import string
from typing import List, Sequence, Tuple

import keras
from keras import backend
from keras.preprocessing.sequence import pad_sequences
from modelforge import Model, register_model
import numpy as np

from sourced.ml.algorithms.id_splitter.nn_model import (f1score, precision,
                                                        recall)
from sourced.ml.models.license import DEFAULT_LICENSE

MAXLEN = 40
PADDING = "post"


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
                  session: "tf.Session" = None, maxlen: int = MAXLEN,
                  padding: str = PADDING) -> "IdentifierSplitterNN":
        assert model is not None

        if session is not None:
            backend.tensorflow_backend.set_session(session)
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

        self._maxlen = maxlen
        self._padding = padding
        self._model = model
        return self

    @property
    def model(self) -> "keras.models.Model":
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
        model = keras.models.Model.from_config(tree["config"])
        model.set_weights(tree["weights"])
        self.construct(model=model)

    def _prepare_single_identifier(self, identifier: str, maxlen: int=MAXLEN, padding: int=PADDING,
                                   mapping: dict=None) -> "np.array":
        if mapping is None:
            mapping = {c: i for i, c in enumerate(string.ascii_lowercase, start=1)}

        # Clean identifier
        clean_id = "".join([char for char in identifier.lower() if char in string.ascii_lowercase])
        if len(clean_id) > MAXLEN:
            clean_id = clean_id[:MAXLEN]
        logging.info("Preprocessed identifier: {}".format(clean_id))
        return np.array([mapping[c] for c in clean_id], dtype="int8"), clean_id

    def prepare_input(self, identifiers: Sequence[str], maxlen: int = MAXLEN,
                      padding: int = PADDING, mapping: dict=None) -> Tuple["np.array", List[str]]:
        """Prepares input by converting a sequence of identifiers to the corresponding
        ascii code 2D-array and the list of lowercase cleaned identifiers.
        """
        processed_ids = []
        clean_ids = []
        for identifier in identifiers:
            feat, clean_id = self._prepare_single_identifier(
                identifier, maxlen=maxlen, padding=padding)
            processed_ids.append(feat)
            clean_ids.append(clean_id)

        processed_ids = pad_sequences(processed_ids, maxlen=maxlen, padding=padding)

        return processed_ids, clean_ids

    def load_model_file(self, path=None):
        assert path is not None
        self._model = keras.models.load_model(path, custom_objects={"precision": precision,
                                                                    "recall": recall,
                                                                    "f1score": f1score})

    def split(self, tokens: Sequence[str]) -> List[str]:
        """
        Splits a lists of tokens using the model.
        """
        feats, clean_ids = self.prepare_input(tokens, maxlen=self._maxlen, padding=self._padding)
        output = self._model.predict(feats, batch_size=4096)
        output = np.round(output)[:, :, 0]
        splitted_ids = []
        for clean_id, id_output in zip(clean_ids, output):
            splitted_id = ""
            for char, label in zip(clean_id, id_output):
                if label == 1:
                    splitted_ids.append(splitted_id)
                    splitted_id = ""
                splitted_id += char
            splitted_ids.append(splitted_id)
        return splitted_ids

    def __del__(self):
        self.clear_keras()
