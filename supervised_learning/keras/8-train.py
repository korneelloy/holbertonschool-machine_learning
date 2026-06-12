#!/usr/bin/env python3
"""training model"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                save_best=False, filepath=None, verbose=True, shuffle=False):
    """
    function that trains a model using mini-batch gradient descent:
    network is the model to train
    data is a numpy.ndarray of shape (m, nx) containing the input data
    labels is a one-hot numpy.ndarray of shape (m, classes)
        containing the labels of data
    batch_size is the size of the batch used for mini-batch gradient descent
    epochs is the number of passes through data for mini-batch gradient descent
    verbose is a boolean that determines if output should be printed
    shuffle is a boolean whether to shuffle the batches every epoch.
    validation_data is the data to validate the model with, if not None
    early_stopping is a boolean that indicates whether early stopping
    early stopping only performed if validation_data exists
    early stopping based on validation loss
    patience is the patience used for early stopping
    learning_rate_decay : boolean whether learning rate decay
    learning rate decay only performed if validation_data exists
    the decay be performed using inverse time decay
    the learning rate decay in a stepwise fashion after each epoch
    each time the learning rate updates, Keras prints a message
    alpha is the initial learning rate
    decay_rate is the decay rate


    save_best is a boolean: save the model after each epoch if best?
    a model is considered the best if validation loss is the lowest
    filepath is the file path where the model should be saved

    Returns: the History object generated after training the model
    """

    def lr_schedule(epoch):
        return alpha / (1 + decay_rate * epoch)

    callbacks = []

    if early_stopping and validation_data:
        early_stop = K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience
        )
        callbacks.append(early_stop)

    if learning_rate_decay and validation_data:
        learning_rate = K.callbacks.LearningRateScheduler(
            lr_schedule, verbose=1)
        callbacks.append(learning_rate)

    if save_best and validation_data:
        save_best_model = K.callbacks.ModelCheckpoint(
            filepath,
            monitor='val_loss',
            verbose=0,
            save_best_only=True,
            mode='auto',
            save_freq='epoch',
            initial_value_threshold=None
        )
        callbacks.append(save_best_model)

    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        validation_data=validation_data,
        shuffle=shuffle,
        callbacks=callbacks,
    )

    return history
