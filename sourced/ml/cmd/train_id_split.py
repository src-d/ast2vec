import argparse
import logging
import os
import pickle


def train_id_split(args: argparse.ArgumentParser):
    """
    Pipeline to train a neural network to split identifiers.

    :param args: arguments.
    """
    from sourced.ml.algorithms.id_splitter.features import prepare_features
    from sourced.ml.algorithms.id_splitter.nn_model import build_rnn, build_cnn, prepare_devices
    from sourced.ml.algorithms.id_splitter.pipeline import make_lr_scheduler, set_random_seed, \
        prepare_callbacks, build_train_generator, create_generator_params, str2ints, \
        config_keras, report

    log = logging.getLogger("train_id_split")
    config_keras()
    set_random_seed(args.seed)

    # prepare features
    X_train, X_test, y_train, y_test = prepare_features(
        csv_path=args.input,
        use_header=args.include_csv_header,
        max_identifier_len=args.length,
        identifier_col=args.csv_identifier,
        split_identifier_col=args.csv_identifier_split,
        test_ratio=args.test_ratio,
        padding=args.padding)

    # prepare train generator
    steps_per_epoch, n_epochs = create_generator_params(
        batch_size=args.batch_size,
        samples_per_epoch=args.samples_before_report,
        n_samples=X_train.shape[0],
        epochs=args.epochs)

    train_gen = build_train_generator(X=X_train, y=y_train, batch_size=args.batch_size)

    # prepare test generator
    validation_steps, _ = create_generator_params(batch_size=args.val_batch_size,
                                                  samples_per_epoch=X_test.shape[0],
                                                  n_samples=X_test.shape[0],
                                                  epochs=args.epochs)
    test_gen = build_train_generator(X=X_test, y=y_test, batch_size=args.val_batch_size)

    # initialize model
    dev0, dev1 = prepare_devices(args.devices)

    # build a RNN model
    if args.model == "RNN":
        model = build_rnn(maxlen=args.length,
                          units=args.neurons,
                          stack=args.stack,
                          optimizer=args.optimizer,
                          rnn_layer=args.type_cell,
                          dev0=dev0,
                          dev1=dev1)

    else:
        # build a CNN model
        filters = str2ints(args.filters)
        kernel_sizes = str2ints(args.kernel_sizes)
        model = build_cnn(maxlen=args.length,
                          filters=filters,
                          output_n_filters=args.dim_reduction,
                          stack=args.stack,
                          kernel_sizes=kernel_sizes,
                          optimizer=args.optimizer,
                          device=dev0)

    log.info("Model summary:")
    model.summary(print_fn=log.info)

    # callbacks
    callbacks = prepare_callbacks(args.output)
    lr_scheduler = make_lr_scheduler(lr=args.lr, final_lr=args.final_lr, n_epochs=n_epochs)
    callbacks = callbacks + (lr_scheduler,)

    # train
    history = model.fit_generator(train_gen, steps_per_epoch=steps_per_epoch,
                                  validation_data=test_gen, validation_steps=validation_steps,
                                  callbacks=list(callbacks), epochs=n_epochs)

    # report quality on test dataset
    report(model, X=X_test, y=y_test, batch_size=args.val_batch_size)

    # save model & history
    # TODO: Use modelforge to save the model
    with open(os.path.join(args.output, "model_history.pickle"), "wb") as f:
        pickle.dump(history.history, f)
    model.save(os.path.join(args.output, "last.model"))
    log.info("Completed!")
