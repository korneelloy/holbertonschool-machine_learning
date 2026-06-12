#!/usr/bin/env python3

import tensorflow as tf
import numpy as np

"""
  function that builds a neural network with the Keras library
  nx: number of input features
  layers: list containing the number of nodes in each layer
  activations: list containing the activation functions
  lambtha: L2 regularization parameter
  keep_prob: probability node will be kept for dropout
  Returns: the keras model
"""


def build_model(nx, layers, activations, lambtha, keep_prob):
    model = tf.keras.Sequential()
    for i in range(len(layers)):
        if i == 0:
            model.add(tf.keras.layers.Dense(
                layers[i],
                activation=activations[i],
                input_shape=(nx,),
                kernel_regularizer=tf.keras.regularizers.l2(lambtha)
            ))
        else:
            model.add(tf.keras.layers.Dense(
                layers[i],
                activation=activations[i],
                kernel_regularizer=tf.keras.regularizers.l2(lambtha)
            ))
        if i < len(layers) - 1:
            model.add(tf.keras.layers.Dropout(1 - keep_prob))
    return model
