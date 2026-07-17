#!/usr/bin/env python3

"""sensitivity confusion matrix"""

import numpy as np


def sensitivity(confusion):
    """
    calculates the sensitivity for each class in a confusion matrix:

    confusion is a confusion numpy.ndarray of shape (classes, classes)
    where row indices represent the correct labels and column indices represent
    the predicted labels
    classes is the number of classes
    Returns: a numpy.ndarray of shape (classes,) containing the sensitivity
    of each class
    sensitivity = tp / (tp + fn)
    """
    true_positives = np.diag(confusion)
    actual_positives = np.sum(confusion, axis=1)

    return true_positives / actual_positives
