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
      Y is a numpy.ndarray with shape (1, m),
        that contains the correct labels for the input data
      A is a numpy.ndarray with shape (1, m),
        containing the activated output of the neuron for each
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

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Setter : Weights vector hidden layer."""
        return self.__W1

    @property
    def b1(self):
        """Setter : Bias hidden layer."""
        return self.__b1

    @property
    def A1(self):
        """Setter : Activated output (prediction) hidden layer."""
        return self.__A1

    @property
    def W2(self):
        """Setter : Weights vector output neuron."""
        return self.__W2

    @property
    def b2(self):
        """Setter : Bias output neuron."""
        return self.__b2

    @property
    def A2(self):
        """Setter : Activated output (prediction) output neuron."""
        return self.__A2

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network
        Updates the private attributes __A1 and __A2
        The neurons should use a sigmoid activation function
        Returns the private attributes __A1 and __A2, respectively
        """
        z1 = np.dot(self.W1, X) + self.b1
        self.__A1 = 1 / (1 + np.exp(-z1))
        z2 = np.dot(self.W2, self.__A1) + self.b2
        self.__A2 = 1 / (1 + np.exp(-z2))

        return (self.__A1, self.__A2)

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        Returns the cost
        """
        cost = -1/Y.shape[1] * \
            np.sum((Y*np.log(A) + (1-Y)*np.log(1.0000001 - A)))
        return cost
