#!/usr/bin/env python3
"""Module defining a single neuron for binary classification."""

import numpy as np


class Neuron:
    """Single neuron performing binary classification.
        @nx : the number of input features to the neuron
        Private instance attributes:
        W: The weights vector for the neuron.
        b: The bias for the neuron.
        A: The activated output of the neuron (prediction)
        X is a numpy.ndarray with shape (nx, m), contains the input data
        m is the number of examples
    """

    def __init__(self, nx):
        """Initialize Neuron with nx input features."""
        if not isinstance(nx, int) or isinstance(nx, bool):
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Setter : Weights vector."""
        return self.__W

    @property
    def b(self):
        """Setter : Bias."""
        return self.__b

    @property
    def A(self):
        """Setter : Activated output (prediction)."""
        return self.__A

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron
        Updates the private attribute __A
        The neuron uses a sigmoid activation function
        Returns the private attribute __A
        """
        z = np.dot(self.W, X) + self.b
        self.__A = 1 / (1 + np.exp(-z))
        return self.__A
