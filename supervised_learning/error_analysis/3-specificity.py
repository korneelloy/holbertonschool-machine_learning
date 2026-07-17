#!/usr/bin/env python3

"""specificity confusion matrix"""

import numpy as np


def specificity(confusion):
    """
    calculates the specificity for each class in a confusion matrix:
    confusion is a confusion numpy.ndarray of shape (classes, classes)
    where row indices represent the correct labels and column indices
    represent the predicted labels
    classes is the number of classes
    Returns: a numpy.ndarray of shape (classes,) containing the
    specificity of each class
    specificity = tn / (tn + fp)
    """

    true_negatives = (np.sum(confusion) - np.sum(confusion, axis=0)
                      - np.sum(confusion, axis=1) + np.diag(confusion))
    false_positives = np.sum(confusion, axis=0) - np.diag(confusion)

    return true_negatives / (true_negatives + false_positives)
