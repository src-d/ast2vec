import logging
import string
import tarfile
from typing import List, Tuple

from modelforge.progress_bar import progress_bar
import numpy


def read_identifiers(csv_path: str, use_header: bool, max_identifier_len: int, identifier_col: int,
                     split_identifier_col: int, shuffle: bool = True) -> List[str]:
    """
    Reads and filters too long identifiers in the CSV file.

    :param csv_path: path to the CSV file.
    :param use_header: uses header as normal line (True) or treat as header line with column names.
    :param max_identifier_len: maximum length of raw identifiers. Skip identifiers that are longer.
    :param identifier_col: column name in the CSV file for the raw identifier.
    :param split_identifier_col: column name in the CSV file for the split identifier lowercase.
    :param shuffle: indicates whether to reorder the list of identifiers
        at random after reading it.
    :return: list of split identifiers.
    """
    log = logging.getLogger("read_identifiers")
    log.info("Reading data from the CSV file %s", csv_path)
    identifiers = []
    # TODO: Update dataset loading as soon as https://github.com/src-d/backlog/issues/1212 done
    # Think about dataset download step
    with tarfile.open(csv_path, encoding="utf-8") as f:
        assert len(f.members) == 1, "One archived file is expected, got: %s" % len(f.members)
        content = f.extractfile(f.members[0])
        if not use_header:
            content.readline()
        for line in progress_bar(content.readlines(), log):
            row = line.decode("utf-8").strip().split(",")
            if len(row[identifier_col]) <= max_identifier_len:
                identifiers.append(row[split_identifier_col])
    if shuffle:
        numpy.random.shuffle(identifiers)
    log.info("Number of identifiers after filtering: %s." % len(identifiers))
    return identifiers


def prepare_features(csv_path: str, use_header: bool, max_identifier_len: int,
                     identifier_col: int, split_identifier_col: int, test_ratio: float,
                     padding: str, shuffle: bool = True) -> Tuple[numpy.array]:
    """
    Prepare the features to train the identifier splitting task.

    :param csv_path: path to the CSV file.
    :param use_header: uses header as normal line (True) or treat as header line with column names.
    :param max_identifier_len: maximum length of raw identifiers. Skip identifiers that are longer.
    :param identifier_col: column in the CSV file for the raw identifier.
    :param split_identifier_col: column in the CSV file for the split identifier.
    :param shuffle: indicates whether to reorder the list of identifiers
        at random after reading it.
    :param test_ratio: Proportion of test samples used for evaluation.
    :param padding: position where to add padding values:
        after the intput sequence if "post", before if "pre".
    :return: training and testing features to train the neural net for the splitting task.
    """
    from keras.preprocessing.sequence import pad_sequences
    log = logging.getLogger("prepare_features")

    # read data from the input file
    identifiers = read_identifiers(csv_path=csv_path, use_header=use_header,
                                   max_identifier_len=max_identifier_len,
                                   identifier_col=identifier_col,
                                   split_identifier_col=split_identifier_col, shuffle=shuffle)

    log.info("Converting identifiers to character indices")
    log.info("Number of identifiers: %d, Average length: %d characters" %
             (len(identifiers), numpy.mean([len(i) for i in identifiers])))

    char2ind = {c: i + 1 for i, c in enumerate(sorted(string.ascii_lowercase))}

    char_id_seq = []
    splits = []
    for identifier in identifiers:
        # iterate through the identifier and convert to array of char indices & boolean split array
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
            elif char == " ":
                split_arr.append(1)
                skip_char = True
            else:
                log.warning("Unexpected symbol %s in identifier", char)
        assert len(index_arr) == len(split_arr)
        char_id_seq.append(index_arr)
        splits.append(split_arr)

    log.info("Number of subtokens: %d, Number of distinct characters: %d" %
             (sum(sum(split_arr) for split_arr in splits) + len(identifiers),
              len({i for index_arr in char_id_seq for i in index_arr})))

    log.info("Train/test splitting...")
    n_train = int((1 - test_ratio) * len(char_id_seq))
    X_train = char_id_seq[:n_train]
    X_test = char_id_seq[n_train:]
    y_train = splits[:n_train]
    y_test = splits[n_train:]
    log.info("Number of train samples: %s, number of test samples: %s" % (len(X_train),
                                                                          len(X_test)))
    log.info("Padding the sequences...")
    X_train = pad_sequences(X_train, maxlen=max_identifier_len, padding=padding)
    X_test = pad_sequences(X_test, maxlen=max_identifier_len, padding=padding)
    y_train = pad_sequences(y_train, maxlen=max_identifier_len, padding=padding)
    y_test = pad_sequences(y_test, maxlen=max_identifier_len, padding=padding)

    return X_train, X_test, y_train[:, :, None], y_test[:, :, None]
