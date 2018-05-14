import argparse
from os.path import dirname, join
import io
import tarfile
import tempfile
import unittest

import numpy as np
from keras.callbacks import TensorBoard, CSVLogger, ModelCheckpoint
from keras.backend.tensorflow_backend import get_session

from sourced.ml.cmd_entries.args import ArgumentDefaultsHelpFormatterNoNone
from sourced.ml.cmd_entries.id_splitter import add_id_splitter_arguments
from sourced.ml.algorithms.id_splitter import prepare_schedule, \
    prepare_callbacks, prepare_devices, prepare_train_generator, to_binary, \
    generator_parameters, prepare_features, read_identifiers, config_keras, register_metric, \
    METRICS, pipeline, prepare_cnn_model, prepare_rnn_model


class Fake:
    pass


def get_identifier_loc():
    _root = dirname(__file__)
    return join(_root, "identifiers.csv.tar.gz")


def write_fake_identifiers(tar_file, n_lines, char_sizes, n_cols, text="a"):
    """
    Prepare file with fake identifiers.
    :param tar_file: ready to write file
    :param n_lines: number of lines to genrate
    :param char_sizes: sizes of identifiers
    :param n_cols: number of columns
    """
    # sanity check
    if isinstance(char_sizes, int):
        char_sizes = [char_sizes] * n_lines
    assert len(char_sizes) == n_lines

    # generate file
    res = []
    for sz in char_sizes:
        line = ",".join([text * sz] * n_cols)
        res.append(line)
    content = "\n".join(res)
    content = content.encode("utf-8")

    # add content to file
    info = tarfile.TarInfo('identifiers.txt')
    info.size = len(content)
    tar_file.addfile(info, io.BytesIO(content))


class IdSplitterTest(unittest.TestCase):
    def test_to_binary(self):
        thresholds = [0, 0.09, 0.19, 0.29, 0.39, 0.49, 0.59, 0.69, 0.79, 0.89, 0.99]
        n_pos = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

        for th, n_p in zip(thresholds, n_pos):
            vals = np.arange(10) / 10
            res = to_binary(vals, th)
            self.assertEqual(sum(to_binary(vals, th)), n_p)
            if th in (0, 0.99):
                self.assertTrue(np.unique(res).shape[0] == 1)
            else:
                self.assertTrue(np.unique(res).shape[0] == 2)

        vals = np.arange(10) / 10
        old_vals = vals.copy()
        for th, n_p in zip(thresholds, n_pos):
            res = to_binary(vals, th, inplace=False)
            self.assertEqual(sum(res), n_p)
            self.assertTrue(np.array_equal(old_vals, vals))
            if th in (0, 0.99):
                self.assertTrue(np.unique(res).shape[0] == 1)
            else:
                self.assertTrue(np.unique(res).shape[0] == 2)

    def test_prepare_devices(self):
        correct_args = ["1", "0,1", "-1"]
        resulted_dev = [("/gpu:1", "/gpu:1"), ("/gpu:0", "/gpu:1"), ("/cpu:0", "/cpu:0")]
        for res, arg in zip(resulted_dev, correct_args):
            args = Fake()
            args.devices = arg
            self.assertEquals(res, prepare_devices(args))

        bad_args = ["", "1,2,3"]
        for arg in bad_args:
            with self.assertRaises(ValueError):
                args = Fake()
                args.devices = arg
                prepare_devices(args)

    def test_prepare_schedule(self):
        start_lr = 10
        end_lr = 1
        n_epochs = 9

        lr_schedule = prepare_schedule(lr=start_lr, final_lr=end_lr, n_epochs=n_epochs)

        for i in range(n_epochs):
            self.assertEqual(start_lr - i, lr_schedule(epoch=i))

        with self.assertRaises(AssertionError):
            lr_schedule(-1)
        with self.assertRaises(AssertionError):
            lr_schedule(n_epochs + 1)

    def test_prepare_train_generator(self):
        batch_size = 3
        # mismatch number of samples
        bad_x = np.zeros(3)
        bad_y = np.zeros(4)
        with self.assertRaises(AssertionError):
            prepare_train_generator(bad_x, bad_y, batch_size=batch_size)

        # check generator with correct inputs
        x = np.zeros(5)
        gen = prepare_train_generator(x, x, batch_size=batch_size)
        expected_n_samples = [3, 2]
        for n_samples in expected_n_samples:
            x_gen, y_gen = next(gen)
            self.assertEquals(x_gen.shape, y_gen.shape)
            self.assertEqual(n_samples, x_gen.shape[0])

    def test_train_parameters(self):
        batch_size = 500
        samples_per_epoch = 10 ** 6
        n_samples = 40 * 10 ** 6
        epochs = 10

        steps_per_epoch_ = samples_per_epoch // batch_size
        n_epochs_ = np.ceil(epochs * n_samples / samples_per_epoch)

        steps_per_epoch, n_epochs = generator_parameters(batch_size, samples_per_epoch, n_samples,
                                                         epochs)
        self.assertEqual(steps_per_epoch, steps_per_epoch_)
        self.assertEqual(n_epochs, n_epochs_)

    def test_config_keras(self):
        config_keras()
        sess = get_session()
        self.assertTrue(sess._config.gpu_options.allow_growth)

    def test_prepare_callbacks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            callbacks = prepare_callbacks(tmpdir)

            # TensorBoard
            self.assertIsInstance(callbacks[0], TensorBoard)
            self.assertTrue(callbacks[0].log_dir.startswith(tmpdir))

            # CSVLogger
            self.assertIsInstance(callbacks[1], CSVLogger)
            self.assertTrue(callbacks[1].filename.startswith(tmpdir))

            # ModelCheckpoint
            self.assertIsInstance(callbacks[2], ModelCheckpoint)
            self.assertTrue(callbacks[2].filepath.startswith(tmpdir))

    def test_register_metric(self):
        fake_metric = "fake metric"
        register_metric(fake_metric)
        self.assertTrue(fake_metric in METRICS)

    def test_raise_register_metric(self):
        bad_metric = 1
        with self.assertRaises(AssertionError):
            register_metric(bad_metric)
        self.assertTrue(bad_metric not in METRICS)

    def test_prepare_features(self):
        # check feature extraction
        text = "a a"
        n_lines = 10
        maxlen = 20
        with tempfile.NamedTemporaryFile() as tmp:
            with tarfile.open(None, "w", fileobj=tmp, encoding="utf-8") as tmp_tar:
                write_fake_identifiers(tmp_tar, n_lines=n_lines, char_sizes=1, n_cols=2, text=text)
            feat = prepare_features(csv_loc=tmp.name, use_header=True, token_col=0, maxlen=maxlen,
                                    mode="r", token_split_col=1, shuffle=True, test_size=0.5,
                                    padding="post")
            x_tr, x_t, y_tr, y_t = feat
            # because of test_size=0.5 - shapes should be equal
            self.assertEqual(x_t.shape, x_tr.shape)
            self.assertEqual(y_t.shape, y_tr.shape)
            # each line contains only one split -> so it should be only 5 nonzero for train/test
            self.assertEqual(np.sum(y_t), 5)
            self.assertEqual(np.sum(y_tr), 5)
            # each line contains only two chars -> so it should be only 10 nonzero for train/test
            self.assertEqual(np.count_nonzero(x_t), 10)
            self.assertEqual(np.count_nonzero(x_tr), 10)
            # y should be 3 dimensional matrix
            self.assertEqual(y_t.ndim, 3)
            self.assertEqual(y_tr.ndim, 3)
            # x should be 2 dimensional matrix
            self.assertEqual(x_t.ndim, 2)
            self.assertEqual(x_tr.ndim, 2)
            # check number of samples
            self.assertEqual(x_t.shape[0] + x_tr.shape[0], n_lines)
            self.assertEqual(y_t.shape[0] + y_tr.shape[0], n_lines)
            # check maxlen
            self.assertEqual(x_t.shape[1], maxlen)
            self.assertEqual(x_tr.shape[1], maxlen)
            self.assertEqual(y_t.shape[1], maxlen)
            self.assertEqual(y_tr.shape[1], maxlen)

        # normal file
        identifiers_loc = get_identifier_loc()
        try:
            prepare_features(csv_loc=identifiers_loc)
        except Exception as e:
            self.fail("prepare_features raised {} with log {}".format(type(e), str(e)))

    def test_read_identifiers(self):
        # read with header
        with tempfile.NamedTemporaryFile() as tmp:
            with tarfile.open(None, "w", fileobj=tmp, encoding="utf-8") as tmp_tar:
                write_fake_identifiers(tmp_tar, n_lines=10, char_sizes=1, n_cols=5)

            res = read_identifiers(csv_loc=tmp.name, use_header=True)
            self.assertEqual(len(res), 10)

        # read without header
        with tempfile.NamedTemporaryFile() as tmp:
            with tarfile.open(None, "w", fileobj=tmp, encoding="utf-8") as tmp_tar:
                write_fake_identifiers(tmp_tar, n_lines=10, char_sizes=1, n_cols=5)

            res = read_identifiers(csv_loc=tmp.name, use_header=False)
            self.assertEqual(len(res), 9)

        # read with maxlen equal to 0 -> expect empty list
        with tempfile.NamedTemporaryFile() as tmp:
            with tarfile.open(None, "w", fileobj=tmp, encoding="utf-8") as tmp_tar:
                write_fake_identifiers(tmp_tar, n_lines=10, char_sizes=1, n_cols=5)

            res = read_identifiers(csv_loc=tmp.name, maxlen=0)
            self.assertEqual(len(res), 0)

        # generate temporary file with identifiers of specific lengths and filter by length
        char_sizes = list(range(1, 11))

        with tempfile.NamedTemporaryFile() as tmp:
            with tarfile.open(None, "w", fileobj=tmp, encoding="utf-8") as tmp_tar:
                write_fake_identifiers(tmp_tar, n_lines=10, char_sizes=char_sizes, n_cols=5)

            # check filtering
            for i in range(11):
                res = read_identifiers(csv_loc=tmp.name, maxlen=i, token_col=3,
                                       token_split_col=4)  # read last two columns as identifiers
                self.assertEqual(len(res), i)

        # read wrong columns
        with tempfile.NamedTemporaryFile() as tmp:
            with tarfile.open(None, "w", fileobj=tmp, encoding="utf-8") as tmp_tar:
                write_fake_identifiers(tmp_tar, n_lines=10, char_sizes=char_sizes, n_cols=2)

            with self.assertRaises(IndexError) as cm:
                read_identifiers(csv_loc=tmp.name, maxlen=10, token_col=3, token_split_col=4)

        # normal file
        identifiers_loc = get_identifier_loc()
        try:
            read_identifiers(csv_loc=identifiers_loc)
        except Exception as e:
            self.fail("read_identifiers raised {} with log {}".format(type(e), str(e)))

    def test_parser(self):
        # normal launch
        try:
            parser = argparse.ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatterNoNone)
            identifiers_loc = "fake_input"
            output_loc = "fake_output"
            add_id_splitter_arguments(parser)
            arguments = "-i {} -o {}".format(identifiers_loc, output_loc)
            args = parser.parse_args(arguments.split())

            self.assertTrue(hasattr(args, "input"))
            self.assertTrue(hasattr(args, "epochs"))
            self.assertTrue(hasattr(args, "batch_size"))
            self.assertTrue(hasattr(args, "length"))
            self.assertTrue(hasattr(args, "output"))
            self.assertTrue(hasattr(args, "test_size"))
            self.assertTrue(hasattr(args, "padding"))
            self.assertTrue(hasattr(args, "optimizer"))
            self.assertTrue(hasattr(args, "rate"))
            self.assertTrue(hasattr(args, "final_rate"))
            self.assertTrue(hasattr(args, "samples_before_report"))
            self.assertTrue(hasattr(args, "val_batch_size"))
            self.assertTrue(hasattr(args, "seed"))
            self.assertTrue(hasattr(args, "num_chars"))
            self.assertTrue(hasattr(args, "devices"))
            self.assertTrue(hasattr(args, "csv_token"))
            self.assertTrue(hasattr(args, "csv_token_split"))
            self.assertTrue(hasattr(args, "csv_header"))

        except Exception as e:
            self.fail("parser raised {} with log {}".format(type(e), str(e)))

        # normal launch RNN
        try:
            parser = argparse.ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatterNoNone)
            identifiers_loc = "fake_input"
            output_loc = "fake_output"
            add_id_splitter_arguments(parser)
            arguments = "-i {} -o {} rnn".format(identifiers_loc, output_loc)
            args = parser.parse_args(arguments.split())

            self.assertTrue(hasattr(args, "input"))
            self.assertTrue(hasattr(args, "epochs"))
            self.assertTrue(hasattr(args, "batch_size"))
            self.assertTrue(hasattr(args, "length"))
            self.assertTrue(hasattr(args, "output"))
            self.assertTrue(hasattr(args, "test_size"))
            self.assertTrue(hasattr(args, "padding"))
            self.assertTrue(hasattr(args, "optimizer"))
            self.assertTrue(hasattr(args, "rate"))
            self.assertTrue(hasattr(args, "final_rate"))
            self.assertTrue(hasattr(args, "samples_before_report"))
            self.assertTrue(hasattr(args, "val_batch_size"))
            self.assertTrue(hasattr(args, "seed"))
            self.assertTrue(hasattr(args, "num_chars"))
            self.assertTrue(hasattr(args, "devices"))
            self.assertTrue(hasattr(args, "csv_token"))
            self.assertTrue(hasattr(args, "csv_token_split"))
            self.assertTrue(hasattr(args, "csv_header"))

            # RNN
            self.assertTrue(hasattr(args, "type"))
            self.assertTrue(hasattr(args, "neurons"))
            self.assertTrue(hasattr(args, "stack"))

        except Exception as e:
            self.fail("parser raised {} with log {}".format(type(e), str(e)))

        # normal launch CNN
        try:
            parser = argparse.ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatterNoNone)
            identifiers_loc = "fake_input"
            output_loc = "fake_output"
            add_id_splitter_arguments(parser)
            arguments = "-i {} -o {} cnn".format(identifiers_loc, output_loc)
            args = parser.parse_args(arguments.split())

            self.assertTrue(hasattr(args, "input"))
            self.assertTrue(hasattr(args, "epochs"))
            self.assertTrue(hasattr(args, "batch_size"))
            self.assertTrue(hasattr(args, "length"))
            self.assertTrue(hasattr(args, "output"))
            self.assertTrue(hasattr(args, "test_size"))
            self.assertTrue(hasattr(args, "padding"))
            self.assertTrue(hasattr(args, "optimizer"))
            self.assertTrue(hasattr(args, "rate"))
            self.assertTrue(hasattr(args, "final_rate"))
            self.assertTrue(hasattr(args, "samples_before_report"))
            self.assertTrue(hasattr(args, "val_batch_size"))
            self.assertTrue(hasattr(args, "seed"))
            self.assertTrue(hasattr(args, "num_chars"))
            self.assertTrue(hasattr(args, "devices"))
            self.assertTrue(hasattr(args, "csv_token"))
            self.assertTrue(hasattr(args, "csv_token_split"))
            self.assertTrue(hasattr(args, "csv_header"))

            # CNN
            self.assertTrue(hasattr(args, "filters"))
            self.assertTrue(hasattr(args, "kernel_sizes"))
            self.assertTrue(hasattr(args, "stack"))
            self.assertTrue(hasattr(args, "dim_reduction"))

        except Exception as e:
            self.fail("parser raised {} with log {}".format(type(e), str(e)))

        # not normal launch - missed output
        with self.assertRaises(SystemExit) as cm:
            parser = argparse.ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatterNoNone)
            identifiers_loc = "fake_input"
            add_id_splitter_arguments(parser)
            arguments = "-i {}".format(identifiers_loc)
            parser.parse_args(arguments.split())

        self.assertEqual(cm.exception.code, 2)

        # not normal launch - missed input
        with self.assertRaises(SystemExit) as cm:
            parser = argparse.ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatterNoNone)
            output_loc = "fake_output"
            add_id_splitter_arguments(parser)
            arguments = "-o {}".format(output_loc)
            parser.parse_args(arguments.split())

        self.assertEqual(cm.exception.code, 2)

    def test_pipeline(self):
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                parser = argparse\
                    .ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatterNoNone)
                output_loc = tmpdir
                identifiers_loc = get_identifier_loc()
                add_id_splitter_arguments(parser)
                arguments = "-i {} -o {} -e 1 --val-batch-size 10 --batch-size 10 --devices -1 " \
                            "--samples-before-report 20 cnn".format(identifiers_loc, output_loc)
                args = parser.parse_args(arguments.split())

                pipeline(args, prepare_model=prepare_cnn_model)
        except Exception as e:
            self.fail("cnn training raised {} with log {}".format(type(e), str(e)))

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                parser = argparse \
                    .ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatterNoNone)
                output_loc = tmpdir
                identifiers_loc = get_identifier_loc()
                add_id_splitter_arguments(parser)
                arguments = "-i {} -o {} -e 1 --val-batch-size 10 --batch-size 10 --devices -1 " \
                            "--samples-before-report 20 rnn".format(identifiers_loc, output_loc)
                args = parser.parse_args(arguments.split())

                pipeline(args, prepare_model=prepare_rnn_model)
        except Exception as e:
            self.fail("rnn training raised {} with log {}".format(type(e), str(e)))


if __name__ == "__main__":
    unittest.main()
