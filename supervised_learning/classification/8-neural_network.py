#!/usr/bin/env python3
"""Module defining a 2 layer neural network for binary classification."""

import numpy as np


class NeuralNetwork:
    """
      nx is the number of input features

      Public instance attributes:
      W1: The weights vector for the hidden layer.
      b1: The bias for the hidden layer.
      A1: The activated output for the hidden layer.
      W2: The weights vector for the output neuron.
      b2: The bias for the output neuron.
      A2: The activated output for the output neuron (prediction).
    """

    def __init__(self, nx, nodes):
        """Initialize NeuralNetwork with nx input features
        and nodes nodes in hidden layer"""

        if not isinstance(nx, int) or isinstance(nx, bool):
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int) or isinstance(nodes, bool):
            raise TypeError("nodes must be an integer")
        elif nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.W1 = np.random.randn(nodes, nx)
        self.b1 = np.zeros((nodes, 1))
        self.A1 = 0
        self.W2 = np.random.randn(1, nodes)
        self.b2 = 0
        self.A2 = 0
