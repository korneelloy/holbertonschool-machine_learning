#!/usr/bin/env python3

"""Batch Normalization"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    creates a batch normalization layer for a neural network in tf:

    prev is the activated output of the previous layer
    n is the number of nodes in the layer to be created
    activation is the activation function that should be used on the output of
    the layer
    you should use the tf.keras.layers.Dense layer as the base layer with
    kernal
    initializer tf.keras.initializers.VarianceScaling(mode='fan_avg')
    your layer should incorporate two trainable parameters, gamma and beta,
    initialized as vectors of 1 and 0 respectively
    you should use an epsilon of 1e-7
    Returns: a tensor of the activated output for the layer
    """

    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    dense = tf.keras.layers.Dense(units=n, kernel_initializer=init)
    Z = dense(prev)

    gamma = tf.Variable(initial_value=tf.ones((1, n)), trainable=True)
    beta = tf.Variable(initial_value=tf.zeros((1, n)), trainable=True)

    mean, variance = tf.nn.moments(Z, axes=[0])
    Z_norm = tf.nn.batch_normalization(Z, mean, variance, beta, gamma, 1e-7)

    return activation(Z_norm)
