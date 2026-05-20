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

    def evaluate(self, X, Y):
        """
        Evaluates the neural network's predictions
        Returns the neuron's prediction and the cost of the network
        The prediction should be a numpy.ndarray with shape (1, m),
            containing the predicted labels for each example
        The label values should be 1 if the output of the network is >= 0.5,
            and 0 otherwise
        """
        A = self.forward_prop(X)[1]
        predictions = np.where(A < 0.5, 0, 1)
        cost = self.cost(Y, A)
        return (predictions, cost)

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network
        alpha is the learning rate
        Updates the private attributes __W1, __b1, __W2, and __b2
        """
        m = Y.shape[1]

        dZ2 = A2 - Y
        dW2 = (1/m) * np.dot(dZ2, A1.T)
        db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

        dZ1 = np.dot(self.W2.T, dZ2) * A1 * (1 - A1)
        dW1 = (1/m) * np.dot(dZ1, X.T)
        db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

        self.__W1 = self.W1 - alpha * dW1
        self.__b1 = self.b1 - alpha * db1
        self.__W2 = self.W2 - alpha * dW2
        self.__b2 = self.b2 - alpha * db2

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Trains the neural network
        alpha is the learning rate
        Returns the evaluation of the training data after iterations
        """

        if not isinstance(iterations, int) or isinstance(iterations, bool):
            raise TypeError("iterations must be an integer")
        elif iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float) or isinstance(alpha, bool):
            raise TypeError("alpha must be a float")
        elif alpha <= 0:
            raise ValueError("alpha must be positive")

        i = 0

        while i < iterations:
            A1, A2 = self.forward_prop(X)
            self.gradient_descent(X, Y, A1, A2, alpha)
            i += 1

        return (self.evaluate(X, Y))
