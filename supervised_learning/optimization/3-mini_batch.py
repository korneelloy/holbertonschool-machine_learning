#!/usr/bin/env python3

"""mini batch"""

import math
import numpy as np
shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """
    creates mini-batches to be used for training a neural network using
    mini-batch gradient descent:
    X is a numpy.ndarray of shape (m, nx) representing input data
    m is the number of data points
    nx is the number of features in X
    Y is a numpy.ndarray of shape (m, ny) representing the labels
    m is the same number of data points as in X
    ny is the number of classes for classification tasks.
    batch_size is the number of data points in a batch
    Returns: list of mini-batches containing tuples (X_batch, Y_batch)
    Your function should allow for a smaller final batch
    (i.e. use the entire dataset)
    """
    X_shuffled, Y_shuffled = shuffle_data(X, Y)

    m = X.shape[0]

    nb_paquet = math.ceil(m / batch_size)

    batches = []

    for i in range(nb_paquet):
        start = i * batch_size
        end = (i + 1) * batch_size
        X_batch = X_shuffled[start:end]
        Y_batch = Y_shuffled[start:end]
        batches.append((X_batch, Y_batch))

    return batches
