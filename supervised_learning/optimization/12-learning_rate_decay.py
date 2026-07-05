#!/usr/bin/env python3

"""Learning Rate Decay"""

import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """
    creates learning rate decay operation in tf using inverse time decay:
    alpha is the original learning rate
    decay_rate: weight used to determine rate at which alpha will decay
    decay_step is the number of passes of gradient descent that should occur
    before alpha is decayed further
    the learning rate decay should occur in a stepwise fashion
    Returns: the learning rate decay operation
    """
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=alpha,
        decay_steps=decay_step,
        decay_rate=decay_rate
    )
    return lr_schedule
