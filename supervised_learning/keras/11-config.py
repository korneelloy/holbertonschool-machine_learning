#!/usr/bin/env python3

"""saving and loading config"""

import tensorflow.keras as K


def save_config(network, filename):
    """
    saves a model's configuration in JSON format:
    network is the model whose configuration should be saved
    filename is the path of the file that the configuration should be saved to
    Returns: None
    """
    network.save(filename, save_format='json')
    return None


def load_config(filename):
    """
    loads a model with a specific configuration:
    filename is the path containing the model's configuration in JSON format
    Returns: the loaded model
    """
    model = K.models.load_model(filename)
    return model
