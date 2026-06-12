#!/usr/bin/env python3
"""training model"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                verbose=True, shuffle=False):
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
    Returns: the History object generated after training the model
    """

    callbacks = []

    if early_stopping and validation_data:
        early_stop = K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience
        )
        callbacks.append(early_stop)

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
