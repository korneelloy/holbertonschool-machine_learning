#!/usr/bin/env python3
"""one-hot matrix"""

import numpy as np


def create_line(classe, classes):
    """transforms label to matrix line"""
    line = []
    for i in range(classes):
        if i == classe:
            line.append(1)
        else:
            line.append(0)
    return line


def one_hot(labels, classes=None):
    """
    function that converts a label vector into a one-hot matrix:
    The last dimension of the one-hot matrix must be the number of classes
    Returns: the one-hot matrix
    """

    if classes is None:
        classes = max(labels) + 1

    matrix = []

    for i in range(len(labels)):
        line = create_line(labels[i], classes)
        matrix.append(line)

    return np.array(matrix)
