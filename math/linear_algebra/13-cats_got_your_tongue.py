#!/usr/bin/env python3


"""concat matrices"""


import numpy as np


def np_cat(mat1, mat2, axis=0):
    """concatenates two matrices along a specific axis"""
    a = np.concatenate((mat1, mat2), axis)
    return a
