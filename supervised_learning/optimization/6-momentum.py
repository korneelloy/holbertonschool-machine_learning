#!/usr/bin/env python3

"""momentum 2"""

import numpy as np
import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """
    sets up gradient descent with momentum optimization algo in TensorFlow
    alpha is the learning rate.
    beta1 is the momentum weight.
    Returns: optimizer
    """

    optimizer = tf.keras.optimizers.SGD(
        learning_rate=alpha,
        momentum=beta1
    )
    return optimizer
