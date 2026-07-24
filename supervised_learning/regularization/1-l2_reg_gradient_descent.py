#!/usr/bin/env python3

"""updates the weights and biases with L2 regularization"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Y is a one-hot numpy.ndarray of shape (classes, m)
    that contains the correct labels for the data
    classes is the number of classes
    mis the number of data points
    weights is a dictionary of the weights and biases of the neural network
    cache is a dictionary of the outputs of each layer of the neural network
    alpha is the learning rate
    lambtha is the L2 regularization parameter
    L is the number of layers of the network
    The neural network uses tanh activations on each layer except the last,
    which uses a softmax activation
    The weights and biases of the network should be updated in place
    """
    m = Y.shape[1]
    dZ = cache['A' + str(L)] - Y

    for x in range(L, 0, -1):
        A_prev = cache['A' + str(x - 1)]
        W = weights['W' + str(x)]

        dW = (1/m) * np.matmul(dZ, A_prev.T) + (lambtha/m) * W
        db = (1/m) * np.sum(dZ, axis=1, keepdims=True)

        if x > 1:
            dZ = np.matmul(W.T, dZ) * (1 - A_prev**2)

        weights['W' + str(x)] = W - alpha * dW
        weights['b' + str(x)] = weights['b' + str(x)] - alpha * db
