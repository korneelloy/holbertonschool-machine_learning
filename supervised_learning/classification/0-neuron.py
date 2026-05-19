#!/usr/bin/env python3

import numpy as np

"""
class Neuron : single neuron performing binary classification
@nx : the number of input features to the neuron
Public instance attributes:
W: The weights vector for the neuron.
b: The bias for the neuron.
A: The activated output of the neuron (prediction)
"""


class Neuron:
    """neuron class for binary classificationin tree"""

    def __init__(self, nx):
        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0

        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")
