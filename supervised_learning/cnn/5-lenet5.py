#!/usr/bin/env python3

"""LeNet-5 (Keras)"""

import numpy as np
from tensorflow import keras as K


def lenet5(X):
    """
    modified version of the LeNet-5 architecture using keras:

    X is a K.Input of shape (m, 28, 28, 1) containing the input images
    m is the number of images

    The model should consist of the following layers in order:
    Convolutional layer with 6 kernels of shape 5x5 with same padding
    Max pooling layer with kernels of shape 2x2 with 2x2 strides
    Convolutional layer with 16 kernels of shape 5x5 with valid padding
    Max pooling layer with kernels of shape 2x2 with 2x2 strides
    Fully connected layer with 120 nodes
    Fully connected layer with 84 nodes
    Fully connected softmax output layer with 10 nodes

    All layers requiring initialization should initialize their kernels
    with the he_normal initialization method

    The seed for the he_normal initializer should be set to zero
    for each layer to ensure reproducibility.

    All hidden layers requiring activation should
    use the relu activation function
    Returns: a K.Model compiled to use Adam optimization
    (with default hyperparameters) and accuracy metrics
    """

    init = K.initializers.he_normal(seed=0)

    conv1 = K.layers.Conv2D(6, (5, 5), padding='same', activation='relu',
                            kernel_initializer=init)(X)
    pool1 = K.layers.MaxPooling2D((2, 2), strides=2)(conv1)
    conv2 = K.layers.Conv2D(16, (5, 5), padding='valid', activation='relu',
                            kernel_initializer=init)(pool1)
    pool2 = K.layers.MaxPooling2D((2, 2), strides=2)(conv2)

    flat = K.layers.Flatten()(pool2)
    fc1 = K.layers.Dense(120, activation='relu',
                         kernel_initializer=init)(flat)
    fc2 = K.layers.Dense(84, activation='relu',
                         kernel_initializer=init)(fc1)
    output = K.layers.Dense(10, activation='softmax',
                            kernel_initializer=init)(fc2)

    model = K.Model(inputs=X, outputs=output)

    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
