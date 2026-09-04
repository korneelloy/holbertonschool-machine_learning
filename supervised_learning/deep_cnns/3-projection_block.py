#!/usr/bin/env python3

"""Projection  Block"""

from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """
    function that builds a projection block as described in
    Deep Residual Learning for Image Recognition (2015):

    A_prev is the output from the previous layer

    filters is a tuple or list containing F11, F3, F12, respectively:
    F11 is the number of filters in the first 1x1 convolution
    F3 is the number of filters in the 3x3 convolution
    F12 is the number of filters in the second 1x1 convolution
    as well as the 1x1 convolution in the shortcut connection

    s is the stride of the first convolution
    in both the main path and the shortcut connection

    All convolutions inside the block should be followed by batch
    normalization along the channels axis
    and a rectified linear activation (ReLU), respectively.

    All weights should use he normal initialization
    The seed for the he_normal initializer should be set to zero
    Returns: the activated output of the projection block

    """

    F11, F3, F12 = filters
    shortcut = A_prev

    # layer 1: 1x1 conv → BN → ReLU
    A_prev = K.layers.Conv2D(
        F11, (1, 1), strides=s,
        kernel_initializer=K.initializers.HeNormal(seed=0))(A_prev)
    A_prev = K.layers.BatchNormalization(axis=3)(A_prev)
    A_prev = K.layers.ReLU()(A_prev)

    # layer 2: 3x3 conv, stride → BN → ReLU
    A_prev = K.layers.Conv2D(
        F3, (3, 3), padding='same',
        kernel_initializer=K.initializers.HeNormal(seed=0))(A_prev)
    A_prev = K.layers.BatchNormalization(axis=3)(A_prev)
    A_prev = K.layers.ReLU()(A_prev)

    # layer 3: 1x1 conv → BN (no ReLU)
    A_prev = K.layers.Conv2D(
        F12, (1, 1),
        kernel_initializer=K.initializers.HeNormal(seed=0))(A_prev)
    A_prev = K.layers.BatchNormalization(axis=3)(A_prev)

    # shortcut: 1x1 conv with stride → BN (no ReLU)
    shortcut = K.layers.Conv2D(
        F12, (1, 1), strides=s,
        kernel_initializer=K.initializers.HeNormal(seed=0))(shortcut)
    shortcut = K.layers.BatchNormalization(axis=3)(shortcut)

    # Add + final ReLU
    A_prev = K.layers.Add()([A_prev, shortcut])
    return K.layers.ReLU()(A_prev)
