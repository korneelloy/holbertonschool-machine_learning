#!/usr/bin/env python3
"""Module defining a single neuron for binary classification."""

import numpy as np


class Neuron:
    """Single neuron performing binary classification.
        @nx : the number of input features to the neuron
        Public instance attributes:
        W: The weights vector for the neuron.
        b: The bias for the neuron.
        A: The activated output of the neuron (prediction)
    """

    def __init__(self, nx):
        """Initialize Neuron with nx input features."""
        if not isinstance(nx, int) or isinstance(nx, bool):
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")

        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0
