#!/usr/bin/env python3

"""precision confusion matrix"""

import numpy as np


def precision(confusion):
    """
    function  that calculates the precision for each class in confusion matrix:

    confusion is a confusion numpy.ndarray of shape (classes, classes)
    where row indices represent the correct labels
    and column indices represent the predicted labels
    classes is the number of classes
    Returns: a numpy.ndarray of shape (classes,) containing the
    precision of each class
    precision   = tp / (tp + fp)
    """
    true_positives = np.diag(confusion)
    actual_positives = np.sum(confusion, axis=0)

    return true_positives / actual_positives
