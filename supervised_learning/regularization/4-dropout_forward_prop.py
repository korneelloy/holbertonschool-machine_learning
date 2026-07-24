#!/usr/bin/env python3

"""forward propagation using Dropout:"""

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    conducts forward propagation using Dropout:

    X is a numpy.ndarray of shape (nx, m) containing the input data for network
    nx is the number of input features
    m is the number of data points
    weights is a dictionary of the weights and biases of the neural network
    L the number of layers in the network
    keep_prob is the probability that a node will be kept
    All layers except the last should use the tanh activation function
    The last layer should use the softmax activation function
    Returns: a dictionary containing the outputs of each layer and the dropout
    mask used on each layer (see example for format)
    """
    cache = {'A0': X}

    for y in range(1, L + 1):
        W = weights['W' + str(y)]
        b = weights['b' + str(y)]
        A_prev = cache['A' + str(y - 1)]
        Z = np.matmul(W, A_prev) + b

        if y == L:
            t = np.exp(Z)
            A = t / np.sum(t, axis=0, keepdims=True)
        else:
            A = np.tanh(Z)
            D = np.random.binomial(1, keep_prob, size=A.shape)
            A = (A * D) / keep_prob
            cache['D' + str(y)] = D

        cache['A' + str(y)] = A

    return cache
